"""Detección visual OPCIONAL Y EXPERIMENTAL (OCR) para eventos que se
muestran como texto en pantalla dentro del juego: la pantalla de "Has
muerto"/"You Died" en Minecraft, texto de eliminación/derrota en shooters,
tu nombre apareciendo en el kill-feed, etc.

Es una capa aparte del audio: no se instala por defecto (usa
`rapidocr-onnxruntime`, importado de forma perezosa) y su precisión depende
mucho de la resolución del video, el idioma del juego y en qué parte de la
pantalla aparece el texto en tu HUD — casi seguro necesita ajuste (qué
palabras buscar, qué región de la pantalla mirar) probando con tu propio
material real. Por eso todo lo relevante (keywords, región, intervalo de
muestreo) es configurable en vez de estar fijo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

# Región = (x, y, w, h) como fracción [0,1] del frame completo.
# None = usar el frame completo (más lento y con más falsos positivos, pero
# no requiere saber de antemano dónde aparece el texto).
Region = tuple[float, float, float, float]


@dataclass
class VisualHit:
    timestamp: float
    text: str
    matched_keyword: str


_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise ImportError(
            "rapidocr-onnxruntime no está instalado. Instálalo con "
            "`pip install rapidocr-onnxruntime` para usar la detección visual, "
            "o desactiva esa opción."
        ) from exc
    _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def sample_frames(video_path: str, interval_seconds: float = 2.5) -> tuple[list[tuple[float, str]], str]:
    """Extrae un frame cada `interval_seconds` con UN SOLO comando ffmpeg
    (mucho más rápido que hacer un seek por cada frame individualmente).
    Devuelve ([(timestamp, ruta_del_frame), ...], carpeta_temporal) — quien
    llama es responsable de borrar la carpeta temporal cuando termine
    (`shutil.rmtree`)."""
    out_dir = tempfile.mkdtemp(prefix="streamclipper_frames_")
    pattern = os.path.join(out_dir, "frame_%06d.jpg")
    fps = 1.0 / interval_seconds
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-qscale:v", "5",
        "-loglevel", "error",
        pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg falló extrayendo frames: {proc.stderr.decode(errors='ignore')}")

    frames = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    return [(i * interval_seconds, os.path.join(out_dir, f)) for i, f in enumerate(frames)], out_dir


def find_text_hits(
    video_path: str,
    keywords: list[str],
    interval_seconds: float = 2.5,
    region: Region | None = None,
    progress_callback=None,
) -> list[VisualHit]:
    """Muestrea frames del video y busca cualquiera de `keywords` (sin
    distinguir mayúsculas/minúsculas ni acentos) en el texto que detecta el
    OCR en cada uno.

    `region` recorta cada frame antes del OCR — más rápido y con menos
    falsos positivos si sabes más o menos dónde aparece el texto (ej. el
    centro de la pantalla para la muerte en Minecraft, la franja superior
    para un kill-feed). Si no se pasa, usa el frame completo.

    `progress_callback(current, total)` opcional, para reportar avance ya
    que esto puede tardar en videos largos (un OCR por cada frame muestreado).

    IMPORTANTE: solo cuenta como "hit" el momento en que una keyword
    APARECE (pasa de no estar en el frame anterior a sí estar en el frame
    actual), no cada frame en el que sigue visible. Esto importa mucho en
    la práctica: algunos overlays (por ejemplo un panel de estadísticas
    tipo "Muertes" en un HUD de stream) muestran la palabra de forma
    prácticamente permanente en pantalla, no solo en el instante de la
    muerte — sin esta lógica, eso generaba un "hit" en CADA frame muestreado
    mientras el panel estuviera visible (cientos de hits en un VOD largo),
    inflando el boost de score de forma continua en vez de marcar momentos
    puntuales. Con la detección de aparición, un overlay permanente genera
    como mucho un hit (cuando aparece por primera vez), mientras que una
    pantalla de muerte real que aparece y desaparece cada vez sigue
    generando un hit por cada muerte, que es el comportamiento que se busca.
    """
    keywords_norm = [_normalize(k) for k in keywords if k.strip()]
    if not keywords_norm:
        return []

    engine = _get_ocr_engine()

    try:
        from PIL import Image
    except ImportError:
        Image = None

    frames, tmp_dir = sample_frames(video_path, interval_seconds=interval_seconds)
    hits: list[VisualHit] = []
    present_prev: set[str] = set()

    try:
        total = len(frames)
        for i, (timestamp, frame_path) in enumerate(frames, start=1):
            image_path = frame_path
            if region and Image is not None:
                try:
                    img = Image.open(frame_path)
                    w, h = img.size
                    rx, ry, rw, rh = region
                    box = (
                        int(rx * w), int(ry * h),
                        int((rx + rw) * w), int((ry + rh) * h),
                    )
                    cropped = img.crop(box)
                    image_path = frame_path + ".crop.jpg"
                    cropped.save(image_path)
                except Exception:
                    image_path = frame_path

            try:
                result, _elapse = engine(image_path)
            except Exception:
                result = None

            present_now: set[str] = set()
            if result:
                text = " ".join(line[1] for line in result)
                text_norm = _normalize(text)
                for kw in keywords_norm:
                    if kw in text_norm:
                        present_now.add(kw)
                        if kw not in present_prev:
                            # Recién apareció desde el frame anterior: esto
                            # es lo que cuenta como el "momento" del evento.
                            hits.append(VisualHit(timestamp=timestamp, text=text, matched_keyword=kw))
            present_prev = present_now

            if progress_callback:
                progress_callback(i, total)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return hits


def _normalize(text: str) -> str:
    """minúsculas + sin tildes, para no perder matches por acentos
    (ej. 'murió' vs 'murio')."""
    import unicodedata
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text
