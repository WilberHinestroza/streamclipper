"""Extracción y análisis de audio a partir de un video, usando solo ffmpeg + numpy.

No depende de librosa/pydub para mantener la instalación ligera. Decodifica el
audio a PCM 16-bit mono a través de un pipe de ffmpeg y calcula una curva de
energía (RMS) por ventanas, que sirve de señal principal para detectar
momentos de "hype" (gritos, risas, reacciones fuertes) en el stream.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000  # Hz, suficiente para energía y para whisper si se usa luego


@dataclass
class EnergyProfile:
    times: np.ndarray  # timestamp (segundos) del centro de cada ventana
    rms: np.ndarray  # energía RMS normalizada [0, 1] por ventana
    window_seconds: float


def extract_pcm(video_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decodifica el audio del video a un array numpy float32 mono en [-1, 1]."""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        "-loglevel", "error",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falló extrayendo audio de {video_path}: {proc.stderr.decode(errors='ignore')}"
        )
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def compute_energy_profile(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
) -> EnergyProfile:
    """Calcula RMS por ventanas deslizantes y lo normaliza a [0, 1]."""
    window = int(window_seconds * sample_rate)
    hop = int(hop_seconds * sample_rate)
    if window <= 0 or hop <= 0:
        raise ValueError("window_seconds y hop_seconds deben ser > 0")

    n_windows = max(0, (len(audio) - window) // hop + 1)
    rms = np.zeros(n_windows, dtype=np.float64)
    times = np.zeros(n_windows, dtype=np.float64)

    for i in range(n_windows):
        start = i * hop
        segment = audio[start:start + window]
        rms[i] = np.sqrt(np.mean(np.square(segment))) if len(segment) else 0.0
        times[i] = (start + window / 2) / sample_rate

    if rms.max() > 0:
        # Normalización robusta: usamos percentil 99 en vez del máximo absoluto
        # para no dejar que un solo pico (ej. un ruido) aplaste toda la escala.
        ceiling = np.percentile(rms, 99) or rms.max()
        rms = np.clip(rms / ceiling, 0.0, 1.0)

    return EnergyProfile(times=times, rms=rms, window_seconds=window_seconds)
