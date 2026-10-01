"""CLI: `python -m streamclipper.cli mi_vod.mp4 --out-dir clips`

Nota sobre las líneas que empiezan con `##`: son marcadores de progreso
pensados para que la interfaz gráfica (gui.py) los lea y actualice una barra
de progreso real. Si corres esto por terminal vas a ver esas líneas también
(son inofensivas), junto con el resto de la salida legible normal.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback

from .audio import compute_energy_profile, extract_pcm
from .clipper import extract_clip, generate_thumbnail, write_temp_srt
from .scoring import apply_keyword_boost, compute_hype_score, find_candidates, merge_overlapping
from .transcribe import (
    find_keyword_timestamps,
    segments_to_srt,
    transcribe,
)
from .facecam import detect_camera_region
from .visual import find_text_hits


def get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(out.stdout.decode().strip())


def emit_progress(**kwargs) -> None:
    """Imprime un marcador de progreso machine-readable para la GUI."""
    print(f"##PROGRESS## {json.dumps(kwargs)}", flush=True)


class _Tee:
    """Escribe a varios streams a la vez (ej. la terminal Y un archivo de
    log), para no tener que elegir entre uno u otro.

    Si un stream no puede codificar algún carácter (ej. la consola de
    Windows en code page cp1252 no puede mostrar ciertos acentos/símbolos
    que salieron de whisper o del OCR), ese error se ignora para ESE
    stream nada más — no debe tirar abajo toda la corrida a mitad de la
    exportación de clips. Ver también el `reconfigure` a UTF-8 al inicio
    de `main()`, que es la corrección principal; esto es solo una red de
    seguridad adicional.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except UnicodeEncodeError:
                try:
                    s.write(data.encode(getattr(s, "encoding", "utf-8") or "utf-8", errors="replace").decode(
                        getattr(s, "encoding", "utf-8") or "utf-8", errors="replace"
                    ))
                    s.flush()
                except Exception:
                    pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def make_run_dir(base_out_dir: str, use_timestamp: bool) -> str:
    """Crea (y devuelve) la carpeta de salida real para esta corrida.

    Si `use_timestamp` es True, crea una subcarpeta con la fecha/hora de esta
    ejecución (ej. `clips/20260829_0911`) para que cada corrida quede en su
    propia carpeta y no se mezclen ni se sobrescriban clips de corridas
    anteriores. Si ya existe una carpeta con ese nombre (dos corridas en el
    mismo minuto), le agrega un sufijo `_2`, `_3`, etc.
    """
    if not use_timestamp:
        os.makedirs(base_out_dir, exist_ok=True)
        return base_out_dir

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    candidate = os.path.join(base_out_dir, stamp)
    suffix = 2
    while os.path.exists(candidate):
        candidate = os.path.join(base_out_dir, f"{stamp}_{suffix}")
        suffix += 1

    os.makedirs(candidate, exist_ok=True)
    return candidate


def suggest_title(transcript_segments, start: float, end: float, max_len: int = 70) -> str | None:
    """Arma un título sugerido a partir del texto transcrito dentro de la
    ventana del clip. No usa ningún modelo/IA extra, solo la transcripción
    de whisper que ya se generó (si --transcribe está activo)."""
    if not transcript_segments:
        return None
    text_parts = [
        s.text.strip() for s in transcript_segments if s.end >= start and s.start <= end and s.text.strip()
    ]
    if not text_parts:
        return None
    text = " ".join(text_parts).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extrae clips destacados de un VOD de stream.")
    p.add_argument("video", help="Ruta al archivo de video del VOD (mp4, mkv, etc.)")
    p.add_argument("--out-dir", default="clips", help="Carpeta base de salida para los clips")
    p.add_argument("--num-clips", type=int, default=8, help="Número máximo de clips a generar")
    p.add_argument("--min-score", type=float, default=0.5, help="Umbral de score [0-1] para considerar un pico")
    p.add_argument("--min-gap", type=float, default=25.0, help="Segundos mínimos entre dos clips")
    p.add_argument("--pre-roll", type=float, default=8.0, help="Segundos antes del pico a incluir")
    p.add_argument("--post-roll", type=float, default=18.0, help="Segundos después del pico a incluir")
    p.add_argument("--vertical", action="store_true", help="Exportar en formato vertical 9:16")
    p.add_argument(
        "--vertical-layout", choices=["crop", "cam-top"], default="crop",
        help="Diseño del formato vertical (requiere --vertical). 'crop' = recorte centrado normal "
             "(default). 'cam-top' = tu cámara en una mitad y el gameplay completo en la otra "
             "(ver --cam-position), detectando la cámara automáticamente o con --cam-region",
    )
    p.add_argument(
        "--cam-region", default="",
        help="Región de tu cámara como 'x,y,w,h' en fracciones 0-1 del frame (ej. '0.02,0.02,0.28,0.35' "
             "para la esquina superior izquierda). Vacío = detectarla automáticamente buscando tu cara "
             "(necesita opencv-python-headless). Solo aplica con --vertical-layout cam-top",
    )
    p.add_argument(
        "--cam-position", choices=["top", "bottom"], default="top",
        help="En qué mitad del vertical va la cámara con --vertical-layout cam-top (default: top)",
    )
    p.add_argument("--fast", action="store_true", help="Corte rápido sin recodificar (-c copy). Más veloz pero el clip puede arrancar en el keyframe más cercano (a veces se ve entrecortado/lento al inicio)")
    p.add_argument("--no-normalize", action="store_true", help="No normalizar el volumen (loudness) de los clips")
    p.add_argument("--no-thumbnails", action="store_true", help="No generar miniaturas .jpg por clip")
    p.add_argument("--no-timestamp", action="store_true", help="No crear subcarpeta con fecha/hora de esta ejecución dentro de --out-dir")
    p.add_argument("--captions", action="store_true", help="Quemar subtítulos (requiere --transcribe)")
    p.add_argument("--transcribe", action="store_true", help="Transcribir con faster-whisper para boost de keywords, subtítulos y título sugerido")
    p.add_argument("--whisper-model", default="base", help="Tamaño del modelo whisper (tiny/base/small/medium/large-v3)")
    p.add_argument("--language", default=None, help="Código de idioma para whisper (ej. es, en). Autodetecta si se omite")
    p.add_argument(
        "--hype-keywords", default="",
        help="Palabras/frases de hype para reforzar el score cuando aparecen en la transcripción, "
             "separadas por coma (ej. 'insano,que locura,vamos'). Vacío = usar la lista por defecto "
             "en español/inglés. Requiere --transcribe",
    )
    p.add_argument(
        "--visual-detect", action="store_true",
        help="EXPERIMENTAL: detección visual por OCR (busca texto en pantalla, ej. 'You Died'). "
             "Mucho más lento — analiza el video frame por frame además del audio.",
    )
    p.add_argument(
        "--visual-keywords", default="",
        help="Palabras a buscar en pantalla, separadas por coma (ej. 'you died,has muerto'). Requiere --visual-detect",
    )
    p.add_argument(
        "--visual-interval", type=float, default=3.0,
        help="Segundos entre cada frame analizado por OCR (default 3.0). Más bajo = más preciso y mucho más lento",
    )
    p.add_argument(
        "--visual-region", default="",
        help="Región de pantalla a mirar como 'x,y,w,h' en fracciones 0-1 (ej. '0.25,0.25,0.5,0.5' para el centro). Vacío = pantalla completa",
    )
    return p


def _make_stream_unicode_safe() -> None:
    """Evita que un carácter raro (típico en texto transcrito por whisper o
    leído por OCR: comillas curvas, acentos combinados, símbolos varios)
    tire abajo todo el programa a mitad de la exportación de clips.

    En Windows, la consola suele usar un code page antiguo (cp1252 u otro)
    que no puede representar todo Unicode — imprimir ahí un carácter fuera
    de ese repertorio lanzaba `UnicodeEncodeError` y el programa se
    detenía de golpe (bug real reportado: se cortó a mitad de exportar el
    clip 7 de 8). Forzamos UTF-8 con reemplazo de caracteres no soportados
    en vez de reventar.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _make_stream_unicode_safe()
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.video):
        print(f"No se encontró el archivo: {args.video}", file=sys.stderr)
        return 1

    out_dir = make_run_dir(args.out_dir, use_timestamp=not args.no_timestamp)
    print(f"##OUTDIR## {out_dir}", flush=True)
    print(f"Carpeta de esta corrida: {out_dir}")

    # A partir de acá, todo lo que se imprima (incluido cualquier error o
    # traceback inesperado) queda guardado también en un log.txt dentro de
    # la carpeta de esta corrida — así no hay que copiar y pegar la salida
    # de la terminal a mano para diagnosticar algo que salió mal.
    log_path = os.path.join(out_dir, "log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)

    try:
        print(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Comando: {' '.join(sys.argv)}")
        print(f"Video: {args.video}")
        print()
        return _run(args, out_dir)
    except Exception:
        print("\nOcurrió un error inesperado y el programa se detuvo:")
        traceback.print_exc()
        return 1
    finally:
        print(f"\n(Este log completo quedó guardado en: {log_path})")
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()


def _run(args: argparse.Namespace, out_dir: str) -> int:
    print(f"[1/6] Leyendo duración de {args.video} ...")
    duration = get_duration(args.video)
    print(f"      Duración: {duration/60:.1f} min")

    print("[2/6] Extrayendo audio y calculando energía ...")
    audio = extract_pcm(args.video)
    profile = compute_energy_profile(audio)

    transcript_segments = []
    keyword_hits: list[float] = []
    if args.transcribe:
        print(f"[3/6] Transcribiendo con whisper ({args.whisper_model}) — esto puede tardar ...")
        try:
            transcript_segments = transcribe(args.video, model_size=args.whisper_model, language=args.language)
            custom_hype_keywords = [k.strip() for k in args.hype_keywords.split(",") if k.strip()] or None
            keyword_hits = find_keyword_timestamps(transcript_segments, keywords=custom_hype_keywords)
            print(f"      {len(transcript_segments)} segmentos, {len(keyword_hits)} con keywords de hype")
            if args.captions and not transcript_segments:
                print(
                    "      Aviso: whisper no encontró texto en el audio (0 segmentos), "
                    "así que no va a haber subtítulos en ningún clip. Puede ser que el "
                    "audio no tenga voz clara, que el idioma no coincida, o que el filtro "
                    "de detección de voz haya descartado todo. Prueba sin --language, o "
                    "con un modelo más grande (--whisper-model small)."
                )
        except ImportError as exc:
            print(f"      Aviso: {exc}")
        except Exception as exc:  # noqa: BLE001
            # La primera vez que se usa un modelo de whisper, se descarga de
            # internet (puede pesar de ~75MB a varios GB según el modelo). Si
            # esa descarga falla (sin internet, firewall, huggingface.co
            # caído, etc.) o hay cualquier otro problema con la transcripción,
            # no tiene sentido tirar todo el programa: seguimos solo con la
            # detección por audio.
            print(f"      Aviso: la transcripción falló y se va a continuar sin ella ({exc})")
    else:
        print("[3/6] Transcripción desactivada (usa --transcribe para activarla)")

    visual_hit_times: list[float] = []
    if args.visual_detect:
        keywords = [k.strip() for k in args.visual_keywords.split(",") if k.strip()]
        if not keywords:
            print("[4/6] Detección visual activada pero sin --visual-keywords, se salta este paso.")
        else:
            region = None
            if args.visual_region:
                try:
                    parts = [float(x) for x in args.visual_region.split(",")]
                    if len(parts) == 4:
                        region = tuple(parts)
                except ValueError:
                    region = None
            print(
                f"[4/6] Escaneando el video buscando texto en pantalla ({', '.join(keywords)}) "
                f"— esto analiza el video frame por frame y puede tardar bastante ..."
            )

            def _visual_progress(current: int, total: int) -> None:
                emit_progress(stage="visual_scan", current=current, total=total)

            try:
                hits = find_text_hits(
                    args.video, keywords,
                    interval_seconds=args.visual_interval,
                    region=region,
                    progress_callback=_visual_progress,
                )
                visual_hit_times = [h.timestamp for h in hits]
                print(f"      {len(hits)} coincidencias encontradas")
                for h in hits[:10]:
                    print(f'        {h.timestamp:.1f}s: "{h.text.strip()}" (keyword: "{h.matched_keyword}")')
                if len(hits) > 10:
                    print(f"        ... y {len(hits) - 10} más")
            except ImportError as exc:
                print(f"      Aviso: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"      Aviso: la detección visual falló y se va a continuar sin ella ({exc})")
    else:
        print("[4/6] Detección visual desactivada (usa --visual-detect para activarla)")

    hype_score = compute_hype_score(profile)
    scores = hype_score
    if keyword_hits:
        scores = apply_keyword_boost(profile.times, scores, keyword_hits, boost=0.4, spread_seconds=3.0)
    if visual_hit_times:
        # Un boost más fuerte que el de keywords de voz: un texto confirmado
        # en pantalla (ej. "You Died") es una señal más directa que una
        # palabra suelta en la transcripción.
        scores = apply_keyword_boost(profile.times, scores, visual_hit_times, boost=0.6, spread_seconds=4.0)

    print("[5/6] Buscando picos de hype y armando clips candidatos ...")
    candidates = find_candidates(
        profile,
        scores=scores,
        min_score=args.min_score,
        min_distance_seconds=args.min_gap,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        video_duration=duration,
        max_candidates=args.num_clips * 3,
    )
    candidates = merge_overlapping(candidates)[: args.num_clips]

    if not candidates:
        emit_progress(stage="done", total=0)
        print("No se encontraron momentos por encima del umbral. Prueba bajando --min-score.")
        return 0

    cam_region = None
    if args.vertical and args.vertical_layout == "cam-top":
        if args.cam_region:
            try:
                parts = [float(x) for x in args.cam_region.split(",")]
                cam_region = tuple(parts) if len(parts) == 4 else None
            except ValueError:
                cam_region = None
            if cam_region is None:
                print(
                    "      Aviso: --cam-region inválido, se esperaba 'x,y,w,h' "
                    "(ej. '0.02,0.02,0.28,0.35'). Se intentará detección automática."
                )
        if cam_region is None:
            print("      Detectando la posición de tu cámara automáticamente ...")
            try:
                cam_region = detect_camera_region(args.video)
                if cam_region:
                    x, y, w, h = cam_region
                    print(f"      Cámara detectada: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f} (fracción de pantalla)")
                else:
                    print(
                        "      Aviso: no se pudo detectar la cámara de forma confiable "
                        "(¿cámara muy pequeña, oculta, o no hay facecam en este video?). "
                        "Usa --cam-region para indicarla a mano; por ahora se usará el "
                        "recorte vertical centrado normal."
                    )
            except ImportError as exc:
                print(f"      Aviso: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"      Aviso: la detección de cámara falló, se usará el recorte vertical centrado normal ({exc})")

    total = len(candidates)
    print(f"[6/6] Exportando {total} clips a {out_dir}/ ...")
    emit_progress(stage="export_start", total=total)

    manifest = []
    for i, cand in enumerate(candidates, start=1):
        out_path = os.path.join(out_dir, f"clip_{i:02d}_score{cand.score:.2f}.mp4")

        srt_path = None
        captions_note = ""
        if args.captions:
            if not transcript_segments:
                captions_note = " [sin subtítulos: no hay transcripción disponible]"
            else:
                clip_segments = [
                    s for s in transcript_segments if s.end >= cand.start and s.start <= cand.end
                ]
                srt_text = segments_to_srt(clip_segments, offset=cand.start)
                if srt_text.strip():
                    srt_path = write_temp_srt(srt_text)
                else:
                    captions_note = " [sin subtítulos: no se transcribió texto en este rango de tiempo]"

        extract_clip(
            args.video, cand, out_path,
            vertical=args.vertical,
            vertical_layout=args.vertical_layout,
            cam_region=cam_region,
            cam_position=args.cam_position,
            srt_path=srt_path,
            reencode=not args.fast,
            normalize_audio=not args.no_normalize,
        )

        thumb_path = None
        if not args.no_thumbnails:
            thumb_path = os.path.splitext(out_path)[0] + ".jpg"
            thumb_offset = min(max(0.0, cand.peak_time - cand.start), (cand.end - cand.start) - 0.2)
            try:
                generate_thumbnail(out_path, thumb_path, timestamp=thumb_offset)
            except RuntimeError as exc:
                print(f"      Aviso: no se pudo generar miniatura para clip {i}: {exc}")
                thumb_path = None

        title = suggest_title(transcript_segments, cand.start, cand.end)
        visual_hit_here = any(cand.start <= t <= cand.end for t in visual_hit_times)

        title_note = f' — "{title}"' if title else ""
        visual_note = " [detección visual]" if visual_hit_here else ""
        print(f"      -> {out_path}  [{cand.start:.1f}s - {cand.end:.1f}s, score={cand.score:.2f}]{title_note}{captions_note}{visual_note}")
        emit_progress(stage="export_progress", current=i, total=total)

        manifest.append({
            "file": out_path,
            "thumbnail": thumb_path,
            "start": round(cand.start, 2),
            "end": round(cand.end, 2),
            "peak_time": round(cand.peak_time, 2),
            "score": round(cand.score, 3),
            "suggested_title": title,
            "captions_burned": srt_path is not None,
            "visual_hit": visual_hit_here,
        })

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    emit_progress(stage="done", total=total)
    print(f"\nListo. Manifest guardado en {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
