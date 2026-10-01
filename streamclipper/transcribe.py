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
import unicodedata
from dataclasses import dataclass

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


def transcribe(video_path: str, model_size: str = "base", language: str | None = None):
    """Transcribe el audio del video. Requiere `pip install faster-whisper`.

    Devuelve una lista de TranscriptSegment. Lanza ImportError con un mensaje
    claro si faster-whisper no está instalado, para que el CLI pueda avisar
    al usuario y seguir sin transcripción en vez de reventar.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper no está instalado. Instálalo con "
            "`pip install faster-whisper` para usar subtítulos y detección "
            "de keywords, o usa --no-transcribe para saltarte este paso."
        ) from exc

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(video_path, language=language, vad_filter=True)

    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments
    ]


def _normalize(text: str) -> str:
    """minúsculas + sin tildes, para no perder matches por acentos
    (ej. 'murió' vs 'murio')."""
    text = text.lower()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def find_keyword_timestamps(
    segments: list[TranscriptSegment],
    keywords: list[str] | None = None,
) -> list[float]:
    """Devuelve los timestamps (inicio del segmento) donde aparece alguna
    keyword de hype, sin distinguir mayúsculas/minúsculas ni acentos.

    IMPORTANTE: usa límites de palabra (no un simple "substring in text"),
    porque una keyword corta como "ace" hacía falso-match dentro de palabras
    comunes en español como "hace" — eso fue un bug real que inflaba mucho
    la cantidad de "hits de hype" en streams hablados en español, contando
    como hype momentos que solo eran conversación normal.
    """
    keywords = keywords or DEFAULT_HYPE_KEYWORDS
    patterns = [
        re.compile(r"(?<!\w)" + re.escape(_normalize(kw)) + r"(?!\w)")
        for kw in keywords
        if kw.strip()
    ]
    hits = []
    for seg in segments:
        text_norm = _normalize(seg.text)
        if any(p.search(text_norm) for p in patterns):
            hits.append(seg.start)
    return hits


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
