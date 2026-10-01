"""Genera un video sintético de prueba: silencio con picos de volumen en
momentos conocidos, para validar que el pipeline detecta esos picos como
clips candidatos. No requiere ningún VOD real.

El audio se sintetiza directamente con numpy (más robusto que encadenar
filtros de ffmpeg) y luego se mezcla con un video de color plano.

Uso: python tests/make_test_video.py test_video.mp4
"""

import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16_000
HYPE_MOMENTS = [15, 45, 90, 140]  # segundos donde metemos un "grito" simulado
TOTAL_DURATION = 180
BURST_SECONDS = 2.0


def build_audio() -> np.ndarray:
    n_samples = int(TOTAL_DURATION * SAMPLE_RATE)
    audio = np.random.normal(0, 0.01, n_samples).astype(np.float32)  # ruido de fondo bajito

    burst_samples = int(BURST_SECONDS * SAMPLE_RATE)
    t = np.linspace(0, BURST_SECONDS, burst_samples, endpoint=False)
    for i, moment in enumerate(HYPE_MOMENTS):
        freq = 300 + i * 40
        tone = 0.9 * np.sin(2 * np.pi * freq * t)
        # envolvente para evitar clicks al inicio/fin del burst
        envelope = np.hanning(burst_samples)
        tone = tone * envelope
        start = int(moment * SAMPLE_RATE)
        end = min(start + burst_samples, n_samples)
        audio[start:end] += tone[: end - start]

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    audio = build_audio()

    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
        wavfile.write(wav_file.name, SAMPLE_RATE, audio)

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s=640x360:d={TOTAL_DURATION}",
            "-i", wav_file.name,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            "-loglevel", "error",
            out_path,
        ]
        subprocess.run(cmd, check=True)

    print(f"Video de prueba creado: {out_path}")
    print(f"Momentos hype esperados (segundos): {HYPE_MOMENTS}")


if __name__ == "__main__":
    main()
