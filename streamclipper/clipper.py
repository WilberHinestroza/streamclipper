"""Extracción de clips de video con ffmpeg a partir de las ventanas de tiempo
detectadas por scoring.py."""

from __future__ import annotations

import os
import subprocess
import tempfile

from .scoring import Candidate


def escape_filter_path(path: str) -> str:
    """Escapa una ruta de archivo para usarla como argumento de un filtro de
    ffmpeg (ej. `subtitles=...`).

    En Windows, una ruta como `C:\\Users\\...\\sub.srt` rompe el parser de
    filtros de ffmpeg de dos formas: el `:` después de la letra de unidad se
    confunde con un separador de opciones, y las barras invertidas `\\` son
    el carácter de escape del propio parser. La forma robusta de evitarlo
    (recomendada en la documentación de ffmpeg) es convertir las barras
    invertidas a barras normales primero — Windows las acepta igual para
    abrir archivos — y luego escapar el `:` que queda.
    """
    normalized = path.replace("\\", "/")
    return normalized.replace(":", r"\:")


def extract_clip(
    video_path: str,
    candidate: Candidate,
    out_path: str,
    vertical: bool = False,
    vertical_layout: str = "crop",
    cam_region: tuple[float, float, float, float] | None = None,
    cam_position: str = "top",
    srt_path: str | None = None,
    reencode: bool = True,
    normalize_audio: bool = True,
) -> None:
    """Corta un clip [candidate.start, candidate.end] del video original.

    Por defecto SIEMPRE recodifica (`reencode=True`) con el `-ss` puesto
    DESPUÉS de `-i`, que es lo que le pide un seek preciso a ffmpeg (busca el
    keyframe más cercano y decodifica desde ahí hasta el punto exacto, sin
    tener que decodificar el archivo completo desde el inicio). Esto evita el
    clásico bug de "el clip arranca en cámara lenta/entrecortado": si se corta
    con `-c copy` y `-ss` antes de `-i`, el corte cae en el keyframe más
    cercano (que puede estar varios segundos antes de lo pedido) y además los
    timestamps del contenedor no quedan en cero, lo que confunde a muchos
    reproductores al arrancar.

    `reencode=False` deja disponible el modo rápido sin recodificar (más
    veloz pero con el riesgo de ese artefacto), por si se necesita velocidad
    por encima de precisión en videos muy largos.

    `vertical_layout="cam-top"` (junto con `cam_region`, una fracción
    `(x, y, w, h)` del frame) arma el vertical con la cámara en una mitad y
    el gameplay completo en la otra, en vez del recorte centrado simple. Si
    `vertical_layout` es `"cam-top"` pero no hay `cam_region`, se cae de
    vuelta al recorte centrado normal (quien llama es responsable de avisar
    de esto al usuario).
    """
    duration = candidate.end - candidate.start

    cmd = ["ffmpeg", "-y"]

    if reencode:
        # -ss como opción de salida (después de -i) = seek preciso.
        cmd += ["-i", video_path, "-ss", f"{candidate.start:.3f}", "-t", f"{duration:.3f}"]
    else:
        # -ss como opción de entrada (antes de -i) = seek rápido pero impreciso.
        cmd += ["-ss", f"{candidate.start:.3f}", "-i", video_path, "-t", f"{duration:.3f}"]

    cmd += ["-loglevel", "error"]

    use_cam_layout = vertical and vertical_layout == "cam-top" and cam_region is not None

    if reencode:
        audio_filters = []
        if normalize_audio:
            # Normaliza el volumen percibido (loudness) a un nivel consistente
            # (-14 LUFS, estándar razonable para redes sociales) para que no
            # tengas clips que suenan más flojos o más fuertes entre sí.
            audio_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")

        if use_cam_layout:
            rx, ry, rw, rh = cam_region
            # Recorta la región de la cámara (en fracción del frame, así no
            # hace falta conocer el tamaño real del video de antemano) y la
            # escala/recorta para llenar la mitad superior o inferior del
            # lienzo vertical 1080x1920 sin deformarla.
            cam_chain = (
                f"[0:v]crop={rw:.4f}*iw:{rh:.4f}*ih:{rx:.4f}*iw:{ry:.4f}*ih,"
                "scale=1080:960:force_original_aspect_ratio=increase,"
                "crop=1080:960,setsar=1[cam]"
            )
            # El gameplay se recorta y escala igual que el modo vertical
            # normal (recorte centrado sobre el frame completo) para llenar
            # la otra mitad. No se "recorta" la cámara fuera del gameplay a
            # propósito: es más simple/robusto y de todos modos la cámara
            # suele ser una ventanita chica en una esquina.
            game_chain = (
                "[0:v]scale=1080:960:force_original_aspect_ratio=increase,"
                "crop=1080:960,setsar=1[game]"
            )
            top_label, bottom_label = ("[cam]", "[game]") if cam_position == "top" else ("[game]", "[cam]")
            final_label = "[vout]"
            stack_chain = f"{top_label}{bottom_label}vstack=inputs=2[vstacked]"
            if srt_path:
                stack_chain += f";[vstacked]subtitles='{escape_filter_path(srt_path)}'{final_label}"
            else:
                final_label = "[vstacked]"

            filter_complex = ";".join([cam_chain, game_chain, stack_chain])
            cmd += ["-filter_complex", filter_complex, "-map", final_label, "-map", "0:a?"]
        else:
            filters = []
            if vertical:
                # Recorta al centro y escala a 1080x1920 (formato TikTok/Shorts/Reels).
                filters.append(
                    "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920"
                )
            if srt_path:
                filters.append(f"subtitles='{escape_filter_path(srt_path)}'")
            if filters:
                cmd += ["-vf", ",".join(filters)]

        if audio_filters:
            cmd += ["-af", ",".join(audio_filters)]

        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-avoid_negative_ts", "make_zero",
        ]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]

    cmd += [out_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló extrayendo clip {out_path}: {proc.stderr.decode(errors='ignore')}"
        )


def generate_thumbnail(clip_path: str, out_path: str, timestamp: float = 1.0) -> None:
    """Extrae un frame del clip ya cortado para usarlo como miniatura."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, timestamp):.3f}",
        "-i", clip_path,
        "-frames:v", "1",
        "-q:v", "3",
        "-loglevel", "error",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló generando miniatura {out_path}: {proc.stderr.decode(errors='ignore')}"
        )


def write_temp_srt(srt_text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return path
