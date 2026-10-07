"""Transcripción opcional con faster-whisper.

Este módulo es opcional a propósito: el pipeline principal (audio.py +
scoring.py) funciona sin él, solo con la energía de audio. Si el usuario
instala `faster-whisper`, se puede usar esto para:
  1. Generar subtítulos (.srt) para quemar en los clips.
  2. Detectar palabras/frases de "hype" y usarlas como boost en el scoring.

Import de faster_whisper es perezoso (lazy) para que el resto del programa
funcione aunque no esté instalado.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from .content import normalize_text as _normalize

# Palabras/frases en español e inglés que suelen indicar un momento de hype
# en streams de videojuegos. Ajustable por el usuario (ver `--hype-keywords`
# en la CLI / el campo de la GUI).
#
# Se sacaron "eso es" y "bro" de la lista por defecto: son demasiado comunes
# como muletillas normales de conversación (no solo en momentos de hype), así
# que generaban muchísimos falsos positivos. Si te sirven para tu forma de
# hablar, puedes agregarlas de vuelta con --hype-keywords.
DEFAULT_HYPE_KEYWORDS = [
    "no way", "no puede ser", "vamos", "let's go", "wtf", "gg", "insano",
    "increíble", "que locura", "no me jodas", "ohh", "wow",
    "clutch", "ace", "headshot", "goool", "gol", "hattrick", "penal",
    "wtf just happened", "hermano no", "cállate",
]


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _load_whisper_model(model_size: str, device: str, compute_type: str):
    """Import perezoso de faster-whisper (para que el resto del programa
    funcione aunque no esté instalado) + construcción del modelo."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper no está instalado. Instálalo con "
            "`pip install faster-whisper` para usar subtítulos y detección "
            "de keywords, o usa --no-transcribe para saltarte este paso."
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _transcribe_with(model, video_path: str, language: str | None) -> list:
    segments, _info = model.transcribe(video_path, language=language, vad_filter=True)
    return list(segments)


def transcribe(video_path: str, model_size: str = "base", language: str | None = None):
    """Transcribe el audio del video, con reintento en CPU si falla la GPU.

    `device="auto"` hace que faster-whisper detecte una GPU NVIDIA y la use,
    pero eso no siempre funciona: si tenés la GPU instalada pero las librerías
    de CUDA de ctranslate2 no están (muy común — alcanza con tener el driver
    instalado, no las runtime libs), la transcripción explota con
    `Library cublas64_12.dll is not found`.

    OJO con el detalle: ese error NO aparece al cargar el modelo (eso sí
    funciona, porque solo reserva memoria), sino recién al transcribir, cuando
    ctranslate2 llama a cuBLAS para la primera vez. Por eso el reintento tiene
    que envolver la transcripción, no solo la carga.

    La CPU SIEMPRE funciona, solo que más lento, así que reintentamos ahí antes
    de dejar al usuario sin subtítulos. Si hubo que caer a CPU se avisa con
    `warnings.warn` para que el CLI lo muestre en vez de perderlo en silencio.

    Devuelve una lista de TranscriptSegment. Lanza ImportError con un mensaje
    claro si faster-whisper no está instalado, para que el CLI pueda avisar
    al usuario y seguir sin transcripción en vez de reventar.
    """
    try:
        model = _load_whisper_model(model_size, device="auto", compute_type="auto")
        segments = _transcribe_with(model, video_path, language)
    except Exception as exc:  # noqa: BLE001
        # No filtramos por tipo de excepción a propósito: si falla la GPU con
        # cualquier cosa (dll faltante, driver incompatible, out of memory),
        # en CPU casi siempre funciona. Solo propagamos el error si el
        # reintento en CPU también falla, y ese error es el que vale la pena
        # mostrar.
        warnings.warn(
            f"falló la transcripción en GPU ({type(exc).__name__}: "
            f"{str(exc).strip().splitlines()[0] if str(exc).strip() else 'sin detalle'}), "
            f"se reintenta en CPU (más lento)",
            RuntimeWarning,
            stacklevel=2,
        )
        model = _load_whisper_model(model_size, device="cpu", compute_type="int8")
        segments = _transcribe_with(model, video_path, language)

    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
    ]


@dataclass
class KeywordHit:
    """Una keyword de hype encontrada en la transcripción.

    `time` es un estimado del momento EXACTO en que se dijo la palabra dentro
    del segmento, no el inicio del segmento entero. Whisper agrupa varias
    frases en un solo segmento (de 2 a 10 segundos es común): usar
    `seg.start` como timestamp centraba el boost y la ventana del clip varios
    segundos antes de la palabra real, y en un segmento largo la palabra
    podía quedar hasta el final — que es justo donde peor queda el recorte.

    La estimación reparte el segmento proporcionalmente a la posición
    (caracteres) de la palabra dentro del texto: no es tan exacto como pedirle
    a whisper los tiempos por palabra (`word_timestamps`), pero no cuesta
    transcribir nada más y es suficiente para anclar la ventana del clip.
    """

    time: float
    keyword: str
    text: str


def _keyword_patterns(keywords: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    pairs: list[tuple[str, re.Pattern[str]]] = []
    for kw in keywords:
        if not kw.strip():
            continue
        pairs.append(
            (
                kw,
                re.compile(r"(?<!\w)" + re.escape(_normalize(kw)) + r"(?!\w)"),
            )
        )
    return pairs


def find_keyword_hits(
    segments: list[TranscriptSegment],
    keywords: list[str] | None = None,
) -> list[KeywordHit]:
    """Devuelve un hit por cada ocurrencia de una keyword de hype, con el
    timestamp estimado de la palabra dentro del segmento.

    IMPORTANTE: usa límites de palabra (no un simple "substring in text"),
    porque una keyword corta como "ace" hacía falso-match dentro de palabras
    comunes en español como "hace" — eso fue un bug real que inflaba mucho
    la cantidad de "hits de hype" en streams hablados en español, contando
    como hype momentos que solo eran conversación normal.

    Antes devolvía solo `seg.start` (un hit por segmento, siempre al inicio);
    ahora hay un hit por ocurrencia, con su propio tiempo estimado.
    """
    keywords = keywords or DEFAULT_HYPE_KEYWORDS
    pairs = _keyword_patterns(keywords)
    hits: list[KeywordHit] = []
    for seg in segments:
        text_norm = _normalize(seg.text)
        if not text_norm:
            continue
        span = float(seg.end) - float(seg.start)
        if span <= 0:
            continue
        for keyword, pattern in pairs:
            for match in pattern.finditer(text_norm):
                fraction = match.start() / len(text_norm)
                hits.append(
                    KeywordHit(
                        time=float(seg.start) + span * fraction,
                        keyword=keyword,
                        text=seg.text,
                    )
                )
    hits.sort(key=lambda h: h.time)
    return hits


def find_keyword_timestamps(
    segments: list[TranscriptSegment],
    keywords: list[str] | None = None,
) -> list[float]:
    """Compat: devuelve solo los timestamps de `find_keyword_hits`."""
    return [h.time for h in find_keyword_hits(segments, keywords)]


def segments_to_srt(segments: list[TranscriptSegment], offset: float = 0.0) -> str:
    """Convierte segmentos de transcripción a formato SRT, con un offset de
    tiempo (para cuando el clip empieza en un punto != 0 del video original)."""

    def fmt(t: float) -> str:
        t = max(0.0, t)
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"

    lines = []
    counter = 1
    for seg in segments:
        start = seg.start - offset
        end = seg.end - offset
        if end < 0:
            continue
        start = max(0.0, start)
        lines.append(str(counter))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(seg.text)
        lines.append("")
        counter += 1
    return "\n".join(lines)
