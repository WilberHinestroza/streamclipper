"""Convierte una curva de energía (+ boosts opcionales de keywords) en una
lista de clips candidatos: picos de "hype" con inicio/fin ya recortados."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import find_peaks

from .audio import EnergyProfile


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
        candidates.append(Candidate(peak_time=peak_time, score=score, start=start, end=end))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def merge_overlapping(candidates: list[Candidate]) -> list[Candidate]:
    """Fusiona clips candidatos cuyas ventanas de tiempo se solapan, quedándose
    con el score más alto y la unión del rango temporal."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.start)
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start <= last.end:
            last.end = max(last.end, current.end)
            if current.score > last.score:
                last.peak_time = current.peak_time
                last.score = current.score
            last.reasons.extend(current.reasons)
        else:
            merged.append(current)
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged
