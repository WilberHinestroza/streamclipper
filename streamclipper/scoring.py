"""Convierte una curva de energía (+ boosts opcionales de keywords) en una
lista de clips candidatos: picos de "hype" con inicio/fin ya recortados."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

from .audio import EnergyProfile
from .content import SegmentLabel, window_evidence


@dataclass
class Candidate:
    peak_time: float
    score: float
    start: float
    end: float
    reasons: list[str] = field(default_factory=list)


def compute_hype_score(
    profile: EnergyProfile,
    baseline_seconds: float = 90.0,
    baseline_percentile: float = 35.0,
) -> np.ndarray:
    """Convierte la energía cruda en un score de "qué tan excepcional es este
    momento COMPARADO CON SU CONTEXTO CERCANO", en vez de compararlo contra
    todo el video de una sola vez.

    Esto importa mucho en la práctica: en un VOD de gameplay, el audio del
    juego (música, disparos, ambiente) suele mantenerse fuerte y bastante
    parejo durante horas. Si se normaliza contra todo el video, ese "piso"
    de ruido constante termina ocupando casi todo el rango de score (ej.
    entre 0.5 y 0.55), y cualquier umbral de sensibilidad por debajo de eso
    agarra prácticamente lo mismo — que es exactamente el bug reportado de
    "la sensibilidad no cambia nada".

    Acá en cambio se calcula, para cada instante, cuál es el nivel de energía
    "típico" de los ~90 segundos alrededor (el percentil 35, no la mediana,
    porque en gameplay el silencio real es raro y el piso de ruido es lo más
    común) y se mide el EXCESO sobre ese nivel local. Un grito real destaca
    incluso con el juego sonando fuerte de fondo, mientras que una sección
    genéricamente ruidosa (pero pareja) casi no genera exceso. El resultado
    se vuelve a normalizar a [0, 1] para que el umbral de sensibilidad sea
    intuitivo.
    """
    if len(profile.rms) == 0:
        return profile.rms.copy()

    hop = profile.times[1] - profile.times[0] if len(profile.times) > 1 else 1.0
    window_samples = max(3, int(round(baseline_seconds / hop)))
    if window_samples % 2 == 0:
        window_samples += 1  # tamaño impar: centra mejor el filtro

    baseline = percentile_filter(
        profile.rms, percentile=baseline_percentile, size=window_samples, mode="nearest"
    )
    excess = np.clip(profile.rms - baseline, 0.0, None)

    ceiling = excess.max()
    if ceiling <= 0:
        return excess  # no hay ningún momento que destaque sobre su entorno

    # Se escala contra el pico más fuerte de TODO el video (no un percentil),
    # para que el momento más "hype" real quede en 1.0 y el resto se ordene
    # proporcionalmente por debajo — con picos que son eventos poco frecuentes
    # (a diferencia del piso de ruido, que es casi todo el video), un percentil
    # alto termina cayendo muy cerca de los picos reales y satura varios de
    # ellos juntos en 1.0 en vez de diferenciarlos.
    return np.clip(excess / ceiling, 0.0, 1.0)


def apply_keyword_boost(
    times: np.ndarray,
    scores: np.ndarray,
    keyword_timestamps: list[float],
    boost: float = 0.4,
    spread_seconds: float = 3.0,
) -> np.ndarray:
    """Suma un boost gaussiano alrededor de los timestamps donde se detectó
    una keyword de hype en la transcripción (ej. "no way", "vamos", "gg").
    Devuelve una copia del array de scores, no modifica el original.

    Toma el MÁXIMO de los boosts gaussianos de cada keyword, no la SUMA.
    Antes se sumaban: si un jugador hablaba mucho y varias keywords caían
    cerca unas de otras (ej. dos o tres en menos de 6 segundos, algo normal
    en alguien que habla rápido), sus boosts de 0.4 se acumulaban entre sí
    (0.4 + 0.4 + 0.4...) y saturaban el score a 1.0 aunque el audio en ese
    momento fuera completamente normal — eso era lo que hacía que, con
    videos muy hablados, terminaran saliendo clips de charla común marcados
    con score=1.00 en vez de los momentos realmente más intensos. Con el
    máximo, una keyword nunca puede aportar más de `boost` en un punto dado,
    sin importar cuántas otras caigan cerca.
    """
    boosted = scores.copy()
    if not keyword_timestamps:
        return boosted
    max_boost = np.zeros_like(scores)
    for t in keyword_timestamps:
        distances = times - t
        gaussian = boost * np.exp(-0.5 * (distances / spread_seconds) ** 2)
        max_boost = np.maximum(max_boost, gaussian)
    return np.clip(boosted + max_boost, 0.0, 1.0)


def find_candidates(
    profile: EnergyProfile,
    scores: np.ndarray | None = None,
    min_score: float = 0.5,
    min_distance_seconds: float = 20.0,
    pre_roll: float = 8.0,
    post_roll: float = 15.0,
    video_duration: float | None = None,
    max_candidates: int = 20,
) -> list[Candidate]:
    """Encuentra picos locales en la curva de score y los convierte en
    ventanas de clip candidatas, ordenadas por score descendente."""
    if scores is None:
        scores = profile.rms

    if len(scores) == 0:
        return []

    hop = profile.times[1] - profile.times[0] if len(profile.times) > 1 else 1.0
    min_distance_samples = max(1, int(min_distance_seconds / hop))

    peak_indices, properties = find_peaks(
        scores,
        height=min_score,
        distance=min_distance_samples,
    )

    candidates = []
    for idx in peak_indices:
        peak_time = float(profile.times[idx])
        score = float(scores[idx])
        start = max(0.0, peak_time - pre_roll)
        end = peak_time + post_roll
        if video_duration is not None:
            end = min(end, video_duration)
        candidates.append(
            Candidate(
                peak_time=peak_time,
                score=score,
                start=start,
                end=end,
                reasons=["pico de energía"],
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def _window_bounds(
    t0: float,
    t1: float,
    video_duration: float | None,
) -> tuple[float, float]:
    start = max(0.0, t0)
    end = t1 if video_duration is None else min(t1, video_duration)
    return start, end


def _window_max(times: np.ndarray, scores: np.ndarray, t0: float, t1: float) -> float:
    """Score máximo dentro de [t0, t1] (o el sample más cercano si no hay
    ninguno exactamente en el rango)."""
    if len(scores) == 0:
        return 0.0
    lo = int(np.searchsorted(times, t0))
    hi = int(np.searchsorted(times, t1))
    if hi <= lo:
        idx = min(max(lo, 0), len(scores) - 1)
        return float(scores[idx])
    return float(scores[lo:hi].max())


def candidates_from_anchors(
    profile: EnergyProfile,
    scores: np.ndarray,
    anchors: list[tuple[float, str]],
    min_score: float = 0.5,
    pre_roll: float = 15.0,
    post_roll: float = 10.0,
    video_duration: float | None = None,
    cluster_gap: float = 8.0,
) -> list[Candidate]:
    """Arma clips candidatos DIRECTAMENTE desde momentos ancla (keywords de la
    transcripción, texto en pantalla), en vez de confiar solo en picos de
    energía.

    Por qué hace falta: los picos de energía y las keywords no caen en el
    mismo instante. Una keyword como "ace", "win" o "derrota" se dice DESPUÉS
    de la jugada (el streamer reacciona al resultado), y find_peaks se queda
    con UN solo máximo local por zona — si el grito quedaba a varios segundos
    de la palabra, la ventana `pico - pre_roll .. pico + post_roll` ponía la
    palabra (y la jugada previa) contra el borde del clip o directamente
    afuera. Ese es el bug de "recortes mal hechos donde no se ve la jugada".

    El recorte de un clip anclado a una keyword es asimétrico a propósito:
    `pre_roll` (antes) es mayor que `post_roll` (después) porque lo que se
    quiere VER es la jugada que produjo la palabra, y esa pasa ANTES de que
    el streamer la diga. La reacción (gritos, "no puede ser") queda cubierta
    por el `post_roll`.

    El score del candidato es el máximo de la curva (con el boost de keywords
    ya aplicado) dentro de la ventana completa: así una keyword dicha en calma
    (score ~0.4 = solo el boost, por debajo de min_score) no genera clip, en
    cambio una keyword durante una jugada intensa sí. Las anclas cercanas
    (misma frase/jugada, `cluster_gap` segundos) se agrupan en un solo
    candidato para no generar ventanas duplicadas.
    """
    if not anchors or len(scores) == 0:
        return []

    ordered = sorted(anchors, key=lambda a: a[0])
    groups: list[list[tuple[float, str]]] = [[ordered[0]]]
    for anchor in ordered[1:]:
        if anchor[0] - groups[-1][-1][0] <= cluster_gap:
            groups[-1].append(anchor)
        else:
            groups.append([anchor])

    candidates: list[Candidate] = []
    for group in groups:
        times_in_group = [t for t, _ in group]
        start, end = _window_bounds(
            min(times_in_group) - pre_roll,
            max(times_in_group) + post_roll,
            video_duration,
        )
        if end <= start:
            continue
        score = _window_max(profile.times, scores, start, end)
        if score < min_score:
            continue
        # El "pico" del candidato es el momento más intenso de la ventana,
        # no la palabra en sí: para el thumbnail y para la separación entre
        # clips interesa el punto de mayor intensidad real.
        window_lo = int(np.searchsorted(profile.times, start))
        window_hi = max(window_lo + 1, int(np.searchsorted(profile.times, end)))
        peak_idx = window_lo + int(np.argmax(scores[window_lo:window_hi]))
        reasons = sorted({reason for _, reason in group})
        candidates.append(
            Candidate(
                peak_time=float(profile.times[min(peak_idx, len(profile.times) - 1)]),
                score=score,
                start=start,
                end=end,
                reasons=reasons,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def grow_candidates(
    candidates: list[Candidate],
    times: np.ndarray,
    scores: np.ndarray,
    max_extend_pre: float = 20.0,
    max_extend_post: float = 12.0,
    floor_ratio: float = 0.35,
    min_floor: float = 0.2,
    max_window: float = 60.0,
    video_duration: float | None = None,
) -> None:
    """Estira cada ventana mientras la actividad (score) se mantenga alta,
    para que el recorte siga la JUGADA y no a un número fijo de segundos.

    Los rolls fijos (`pre_roll`/`post_roll`) tienen un dilema: si son
    conservadores, cortan la jugada; si son largos, agregan relleno muerto.
    Esta función los toma como punto de partida y después extiende el borde
    mientras el score se mantenga por encima de un piso (`floor_ratio` del
    score del propio candidato, con un mínimo absoluto): la acción
    sostenida (una ronda, un push, una discusión) queda entera, y el clip
    se corta en cuanto vuelve la calma.

    Límites: `max_extend_pre`/`max_extend_post` son el alargue máximo por
    lado y `max_window` la duración total máxima resultante, para que un
    tramo con score alto sostenido durante minutos no termine en un clip de
    varios minutos.

    Modifica los candidatos in place.
    """
    if not candidates or len(scores) == 0 or len(times) < 2:
        return

    for cand in candidates:
        floor = max(min_floor, floor_ratio * cand.score)

        # Hacia adelante: camina muestra por muestra desde el borde derecho
        # mientras siga alto y no se pase de los límites.
        j = int(np.searchsorted(times, cand.end))
        while (
            j < len(times)
            and scores[j] >= floor
            and times[j] <= cand.end + max_extend_post
            and times[j] - cand.start <= max_window
        ):
            j += 1
        if j > 0:
            cand.end = max(cand.end, float(times[j - 1]))

        # Hacia atrás: lo mismo desde el borde izquierdo, ya con el fin
        # definitivo (así `max_window` limita la duración total, no cada
        # extensión por separado).
        i = int(np.searchsorted(times, cand.start))
        while (
            i > 0
            and scores[i - 1] >= floor
            and times[i - 1] >= cand.start - max_extend_pre
            and cand.end - times[i - 1] <= max_window
        ):
            i -= 1
        cand.start = min(cand.start, max(0.0, float(times[i])))

        if video_duration is not None:
            cand.end = min(cand.end, video_duration)
            cand.start = max(0.0, min(cand.start, cand.end))


def apply_chat_penalty(
    candidates: list[Candidate],
    labels: list[SegmentLabel],
    visual_times: list[float],
    stream_kind: str,
    penalty: float = 0.6,
    peak_context: float = 8.0,
    min_chat_seconds: float = 1.0,
) -> int:
    """Resta `penalty` a los candidatos cuyo pico es, por evidencia del
    transcript, una interacción con el chat y NO tiene nada de juego.

    El problema que resuelve: la energía de audio no distingue un grito por
    una jugada de un grito porque el streamer está peleando con el chat —
    al final del día son el mismo volumen. La transcripción sí lo distingue:
    si dentro del clip no aparece NINGÚN término de juego (ni texto en
    pantalla detectado por OCR) y sí hay medio segundo o más de charla con el
    chat justo alrededor del pico, el grito es con el chat y se penaliza.

    Decisiones de diseño importantes:

    - Solo se penaliza con evidencia de chat Y cero evidencia de juego. Si el
      clip es mixto o no hay transcripción dentro, se conserva (ante la duda,
      no se descarta un clip bueno).
    - En `stream_kind == "justchatting"` no se penaliza nada: si el streamer
      no está jugando, pelear con el chat ES el contenido.
    - El descarte final lo hace quien llama (vuelve a filtrar por min_score),
      así que con `penalty=0` esta función no hace nada y con un penalty
      chico solo degrada el ranking en vez de tirar el clip.

    Devuelve la cantidad de candidatos penalizados.
    """
    if penalty <= 0 or not candidates or not labels:
        return 0
    if stream_kind == "justchatting":
        return 0

    penalized = 0
    for cand in candidates:
        evidence = window_evidence(labels, cand.start, cand.end)
        if evidence.game_hits > 0:
            continue
        if any(cand.start <= t <= cand.end for t in visual_times):
            continue
        near = window_evidence(
            labels,
            max(cand.start, cand.peak_time - peak_context),
            min(cand.end, cand.peak_time + peak_context),
        )
        if near.chat_hits <= 0 or near.chat_seconds < min_chat_seconds:
            continue
        cand.score = max(0.0, cand.score - penalty)
        cand.reasons.append(f"charla con el chat (-{penalty:g})")
        penalized += 1
    return penalized


def select_best(
    candidates: list[Candidate],
    max_clips: int,
    min_score: float = 0.0,
    min_gap: float = 0.0,
    overlap_tolerance: float = 5.0,
) -> list[Candidate]:
    """Selección final: ordena por score y va tomando mientras no se pisen
    con uno ya elegido (ventanas solapadas ni picos más cerca de `min_gap`).

    Hasta ahora ese trabajo lo hacía find_peaks (por distancia entre picos) y
    merge_overlapping (por solape), pero con las ventanas crecidas y los
    candidatos anclados a keywords hay más formas de que dos clips se
    superpongan — conviene validarlo acá, sobre los candidatos finales.

    `overlap_tolerance` deja pasar solapes de hasta ese tamaño (segundos):
    dos momentos distintos que quedan a segundos de distancia se llevan peor
    un clip compartido que dos clips con un segundo o dos repetidos. El
    criterio es espejo del `min_overlap` de `merge_overlapping`: los solapes
    de ese tamaño o más ya quedaron fusionados antes de llegar acá.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: list[Candidate] = []
    for cand in ordered:
        if cand.score < min_score:
            continue
        if len(selected) >= max_clips:
            break
        conflicts = False
        for other in selected:
            overlap = min(cand.end, other.end) - max(cand.start, other.start)
            too_close = abs(cand.peak_time - other.peak_time) < min_gap
            if overlap > overlap_tolerance or too_close:
                conflicts = True
                break
        if not conflicts:
            selected.append(cand)
    return selected


def merge_overlapping(
    candidates: list[Candidate],
    max_span: float | None = None,
    min_overlap: float = 0.0,
) -> list[Candidate]:
    """Fusiona clips candidatos cuyas ventanas de tiempo se solapan, quedándose
    con el score más alto y la unión del rango temporal.

    Dos parámetros evitan los excesos de esta fusión:

    - `max_span` limita cuánto puede crecer la unión: sin él, una seguidilla
      de ventanas encadenadas (keywords cada pocos segundos en una sección
      muy hablada) podría fusionarse en UN clip gigante que cubre media hora.
      Si la unión resultante superaría ese largo, los candidatos quedan
      separados.
    - `min_overlap` exige que el solape sea de al menos ese número de
      segundos para fusionar. Sin él, dos momentos DISTINTOS que quedan a
      pocos segundos (el pico de energía de uno y la ventana anclada a la
      keyword del otro, por ejemplo) se fusionaban en un clip largo con los
      dos adentro, en vez de dar un clip por momento.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.start)
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        union_end = max(last.end, current.end)
        overlap = last.end - current.start
        union_too_long = max_span is not None and union_end - last.start > max_span
        if overlap >= min_overlap and not union_too_long:
            last.end = union_end
            if current.score > last.score:
                last.peak_time = current.peak_time
                last.score = current.score
            last.reasons.extend(current.reasons)
        else:
            merged.append(current)
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged
