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


def _subtitle_filter(srt_path: str, margin: int | None = None) -> str:
    """Filtro de subtítulos quemados, opcionalmente subido.

    Sin `margin` se usa la posición por defecto (abajo del todo). Con `margin`
    se sube esa cantidad de píxeles desde el borde inferior, para que en el
    vertical con fondo desenfocado caigan SOBRE el gameplay y no sobre el
    fondo borroso. El valor lo calcula `subtitle_margin_v` según el aspecto
    real del video de origen.

    OJO con los timestamps del .srt: como el `-ss` va como opción de SALIDA
    (después de `-i`), ffmpeg NO hace seek en el grafo de filtros — decodifica
    desde el inicio del video y descarta frames hasta el punto pedido. El filtro
    `subtitles` ve entonces la línea de tiempo COMPLETA del video (0, 0.5, 1.0,
    ... hasta el final), no la del clip. Por eso el .srt tiene que llevar
    timestamps ABSOLUTOS (809.5, 810.0, ...); con timestamps relativos al clip
    (0..26s) los cues se dibujan en frames que después se descartan y el
    subtítulo no aparece nunca. Salvo en el clip que empieza en 0, donde por
    casualidad coincidían. Lo arma el CLI con `segments_to_srt(..., offset=0)`.
    """
    if margin:
        return (
            f"subtitles='{escape_filter_path(srt_path)}'"
            f":force_style='Alignment=2,MarginV={int(margin)}'"
        )
    return f"subtitles='{escape_filter_path(srt_path)}'"


def _fit_over_blur_chain(
    w: int,
    h: int,
    blur_sigma: int = 12,
    cam_region: tuple[float, float, float, float] | None = None,
    out_label: str = "vout",
) -> str:
    """Construye el filtro que mete el frame COMPLETO dentro de un lienzo
    `w` x `h` (9:16) sin recortar el gameplay, con una copia de fondo
    ampliada y desenfocada ocupando el espacio sobrante.

    Es la diferencia entre "se ve mal porque recorta" y "se ve entero":
    antes el filtro hacía `crop` centrado, que en un gameplay 16:9 se
    quedaba con solo el ~32% del ancho del juego. Acá la capa de juego usa
    `force_original_aspect_ratio=decrease` (cabe entero) y la de fondo usa
    `increase` (cubre todo) + desenfoque.

    El fondo NO usa un `gblur` fuerte y caro: primero se reduce a 1/4 de
    resolución y después se vuelve a escalar a `w`x`h`. Ese reescalado hacia
    arriba es lo que produce el desenfoque, y sale mucho más barato que
   difuminar un frame de 1080x1920 píxel a píxel.

    `cam_region` es opcional: si viene, la capa de juego es un recorte de esa
    región (usado por el layout 'cam-top'), y el fondo se arma con el frame
    entero para que el relleno quede lleno.
    """
    # El fondo se reduce a 1/4 antes de difuminar: el upscale posterior hace
    # el resto del trabajo y el coste de gblur cae ~16x.
    bw, bh = max(2, w // 4), max(2, h // 4)
    parts = [
        f"[0:v]split=2[bgsrc][fgsrc]",
        f"[bgsrc]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
        f"crop={bw}:{bh},scale={w}:{h},setsar=1,gblur=sigma={blur_sigma}[bg]",
    ]
    if cam_region:
        rx, ry, rw, rh = cam_region
        parts.append(
            f"[fgsrc]crop={rw:.4f}*iw:{rh:.4f}*ih:{rx:.4f}*iw:{ry:.4f}*ih,"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fg]"
        )
    else:
        parts.append(
            f"[fgsrc]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fg]"
        )
    parts.append(f"[bg][fg]overlay=(W-w)/2:(H-h)/2[{out_label}]")
    return ";".join(parts)


def crop_center_chain(w: int, h: int, out_label: str = "vout") -> str:
    """El layout vertical ORIGINAL: recorte centrado al 9:16.

    Se conserva por compatibilidad (--vertical-fit crop), pero crops el
    gameplay sin piedad: en 16:9 sobre un lienzo 9:16 se queda con el centro
    y se pierde la mayor parte de la pantalla del juego. Para eso está
    `fit_over_blur_chain`, que es el default ahora.
    """
    return (
        f"[0:v]crop='min(iw,ih*{w}/{h})':'min(ih,iw*{h}/{w})',"
        f"scale={w}:{h},setsar=1[{out_label}]"
    )


def subtitle_margin_v(
    video_width: int,
    video_height: int,
    target_w: int = 1080,
    target_h: int = 1920,
    pad_above_game_edge: int = 170,
) -> int:
    """Calcula el `MarginV` para los subtítulos del vertical, de modo que caigan
    SOBRE el gameplay y no sobre el fondo desenfocado de abajo.

    El gameplay entra completo con `decrease`, así que su alto real en el
    lienzo es `video_height * min(target_w/video_width, target_h/video_height)`
    y queda centrado verticalmente: el borde inferior del gameplay queda en
    `target_h - (target_h - game_h) / 2`, y `gap` es el hueco de fondo que hay
    entre ese borde y el final del lienzo.

    Para que el texto no caiga en ese hueco hay que moverlo hacia arriba unos
    `gap + pad` píxeles. Y acá está el detalle contraintuitivo: `MarginV`
    (que en ASS es la distancia al borde INFERIOR) NO mueve el texto en esa
    dirección — medido empíricamente sobre un lienzo de 1920, CADA unidad de
    MarginV sube el texto 6.4px. Por eso el desplazamiento deseado se divide
    por ese factor.
    """
    if video_width <= 0 or video_height <= 0:
        return 0
    scale = min(target_w / video_width, target_h / video_height)
    game_h = video_height * scale
    game_bottom = target_h - (target_h - game_h) / 2.0

    # Un video ya vertical tiene el juego pegado al borde inferior y no hay
    # fondo que tapar: no hace falta mover nada (y moverlo sacaría el texto
    # de cuadro).
    gap = target_h - game_bottom
    if gap < 10:
        return 0

    wanted = gap + pad_above_game_edge
    # ASS_SSRT_SCALE: factor con el que libass convierte MarginV a píxeles reales
    # (medido en ffmpeg 9 sobre un lienzo de 1920: 6.4 px por unidad).
    scale_factor = target_h / 300.0
    return max(0, int(round(wanted / scale_factor)))


def probe_dimensions(video_path: str) -> tuple[int, int]:
    """Lee el ancho y alto del video con ffprobe, para poder calcular cosas
    que dependen del aspecto (ej. la posición de los subtítulos en vertical).

    Si ffprobe falla o no devuelve dimensiones usable, devuelve (0, 0) en vez
    de cortar la corrida: los callers tratan eso como "no sé el aspecto" y
    usan los defaults.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", video_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            return (0, 0)
        out = proc.stdout.decode().strip().splitlines()[0]
        parts = out.split("x")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


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
    vertical_fit: str = "blur",
    subtitle_margin: int | None = None,
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
            # La cámara va a media pantalla y el gameplay a la otra, cada uno
            # en su propio lienzo 1080x960. Para el gameplay usamos la misma
            # lógica "fit sobre fondo desenfocado" que en el vertical normal,
            # pero contra un lienzo 1080x960: así el gameplay entra COMPLETO
            # en su mitad, en vez de recortarse al centro (que en 16:9
            # empujaba la mitad del juego fuera de cuadro).
            cam_chain = (
                f"[0:v]crop={rw:.4f}*iw:{rh:.4f}*ih:{rx:.4f}*iw:{ry:.4f}*ih,"
                "scale=1080:960:force_original_aspect_ratio=increase,"
                "crop=1080:960,setsar=1[cam]"
            )
            game_chain = _fit_over_blur_chain(
                1080, 960, out_label="game"
            )
            top_label, bottom_label = ("[cam]", "[game]") if cam_position == "top" else ("[game]", "[cam]")
            final_label = "[vout]"
            stack_chain = f"{top_label}{bottom_label}vstack=inputs=2[vstacked]"
            if srt_path:
                stack_chain += f";[vstacked]{_subtitle_filter(srt_path, subtitle_margin)}[vout]"
            else:
                final_label = "[vstacked]"

            filter_complex = ";".join([cam_chain, game_chain, stack_chain])
            cmd += ["-filter_complex", filter_complex, "-map", final_label, "-map", "0:a?"]
        elif vertical:
            # Vertical 9:16. El default es 'blur': el gameplay entra entero
            # con un fondo desenfocado que llena el resto del 9:16 (sin
            # perder nada de la pantalla del juego, que era el defecto del
            # crop centrado). 'crop' conserva el recorte al centro de antes.
            if vertical_fit == "crop":
                video_chain = crop_center_chain(1080, 1920, out_label="vout")
            else:
                video_chain = _fit_over_blur_chain(1080, 1920, out_label="vout")
            if srt_path:
                sub = _subtitle_filter(srt_path, subtitle_margin)
                video_chain = f"{video_chain};[vout]{sub}[vsub]"
                final_label = "[vsub]"
            else:
                final_label = "[vout]"
            cmd += ["-filter_complex", video_chain, "-map", final_label, "-map", "0:a?"]
        else:
            filters = []
            if srt_path:
                filters.append(_subtitle_filter(srt_path, subtitle_margin))
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
