"""Test de integración del CLI con la transcripción SIMULADA.

No hace falta hablar ni tener un VOD: se parchea `cli.transcribe` para que
devuelva segmentos inventados encima del video sintético de
`tests/make_test_video.py` (picos de energía en 15, 45, 90 y 140 segundos) y
se corre el pipeline completo (con ffmpeg incluido) para verificar los dos
comportamientos reportados:

  1. una keyword dicha DESPUÉS de la jugada ("ace" en t=46) genera un clip
     que SÍ incluye la jugada (el pico de energía en t=45), y
  2. gritos cuyo transcript es solo charla con el chat (t=90 y t=140) quedan
     fuera de los resultados en modo gameplay, pero entran si el stream es
     Just Chatting.

Corre de forma standalone (tarda unos minutos porque exporta clips reales):

    python tests/test_cli_stubbed.py
"""

import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from streamclipper import cli
from streamclipper.transcribe import TranscriptSegment

VIDEO = os.path.join(PROJECT_ROOT, "test_video.mp4")

# Transcripción simulada sobre el video sintético:
#  - ~45s: jugada de JUEGO que termina en la keyword "ace" (la keyword viene
#    después de la acción, como en la vida real).
#  - ~90s: pico de energía pero charlando/peleando con el chat -> penalizado.
#  - ~140s: pico de energía y charla con el chat también (pero hay keyword).
FAKE_SEGMENTS = [
    TranscriptSegment(36.0, 41.0, "ok chicos vamos la ronda"),
    TranscriptSegment(41.0, 44.5, "viene el enemigo, lo veo"),
    TranscriptSegment(44.5, 47.0, "gg ace"),
    TranscriptSegment(84.0, 90.0, "chat, ustedes dicen que siempre pierdo"),
    TranscriptSegment(90.0, 96.0, "no way, no me digan eso"),
    TranscriptSegment(134.0, 140.0, "buenas noches gente del chat"),
    TranscriptSegment(140.0, 146.0, "vamos, suscríbete si te gusta"),
]


def fake_transcribe(video_path, model_size="base", language=None):
    return list(FAKE_SEGMENTS)


def ensure_test_video() -> None:
    if os.path.isfile(VIDEO):
        return
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "tests", "make_test_video.py"), VIDEO],
        check=True,
        cwd=PROJECT_ROOT,
    )


_RUN_CACHE: dict[tuple, list[dict]] = {}


def run_cli(extra_args: list[str]) -> list[dict]:
    key = tuple(extra_args)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    out_dir = tempfile.mkdtemp(prefix="streamclipper_test_")
    original = cli.transcribe
    cli.transcribe = fake_transcribe
    try:
        rc = cli.main(
            [
                VIDEO,
                "--out-dir", out_dir,
                "--transcribe",
                "--min-score", "0.5",
                "--num-clips", "8",
                "--no-timestamp",
                "--no-thumbnails",
                "--no-normalize",
            ]
            + list(extra_args)
        )
    finally:
        cli.transcribe = original
    assert rc == 0, f"el CLI terminó con código {rc}"
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    _RUN_CACHE[key] = manifest
    return manifest


def covers(manifest: list[dict], moment: float) -> bool:
    return any(m["start"] <= moment <= m["end"] for m in manifest)


def test_keyword_after_the_play_covers_the_play():
    manifest = run_cli([])
    # El clip debe empezar ANTES de la jugada (pico en 45) aunque la keyword
    # "ace" se diga recién en ~46.
    clip = [m for m in manifest if m["start"] <= 45.0 <= m["end"]]
    assert clip, f"no hay ningún clip cubriendo la jugada de 45s: {manifest}"
    assert clip[0]["start"] <= 42.0, f"el clip corta la jugada: {clip[0]}"
    assert any("keyword" in r for r in clip[0]["reasons"]), clip[0]["reasons"]

    # Los picos sin transcript asociado siguen saliendo.
    assert covers(manifest, 15.0), manifest


def test_chat_only_screams_are_dropped_in_gameplay():
    manifest = run_cli([])
    assert not covers(manifest, 90.0), f"se coló un clip de charla con el chat: {manifest}"
    assert not covers(manifest, 140.0), f"se coló un clip de charla con el chat: {manifest}"


def test_chat_screams_are_kept_in_justchatting():
    manifest = run_cli(["--stream-kind", "justchatting"])
    assert covers(manifest, 90.0), manifest
    assert covers(manifest, 140.0), manifest


def _run_all():
    ensure_test_video()
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"OK  {name}")
    print(f"\n{len(tests)} tests OK")


if __name__ == "__main__":
    _run_all()
