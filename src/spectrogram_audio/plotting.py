"""Optional diagnostic plots for the spectrogram codec.

This module requires the ``plot`` extra (matplotlib).  The core codec never
imports it, so a minimal installation stays small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import SpectrogramAudioError


def plot_diagnostics(
    db_spectrogram: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_length: int,
    freq_max: float,
    duration: float,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Write a three-panel figure comparing the image and the reconstruction.

    Args:
        db_spectrogram: Decoded spectrogram (row 0 at 0 Hz).
        audio: Reconstructed waveform.
        sample_rate: Sample rate of ``audio``.
        n_fft: FFT size used by Griffin-Lim.
        hop_length: Hop length used by Griffin-Lim.
        freq_max: Highest encoded frequency in hertz.
        duration: Original duration in seconds.
        output_path: Destination image path.
        dpi: Figure resolution.

    Returns:
        The path of the written figure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import librosa
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SpectrogramAudioError(
            "matplotlib is required for diagnostic plots; install it with: uv sync --extra plot"
        ) from exc

    db = np.asarray(db_spectrogram, dtype=np.float32)
    y = np.asarray(audio, dtype=np.float32)
    vmin = float(db.min())
    vmax = float(db.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Spectrogram audio codec diagnostics", fontsize=13)

    encoded = axes[0].imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration, 0.0, freq_max),
        cmap="gray_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Spectrogram recovered from the image")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(encoded, ax=axes[0], label="Level (dB)")

    time_axis = np.arange(y.size, dtype=np.float64) / sample_rate
    axes[1].plot(time_axis, y, color="steelblue", linewidth=0.7)
    axes[1].set_title("Reconstructed waveform")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlim(0.0, max(duration, float(time_axis[-1]) if time_axis.size else 0.0))
    axes[1].grid(alpha=0.25)

    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    reconstructed_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    panel = axes[2].imshow(
        reconstructed_db,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration, 0.0, freq_max),
        cmap="magma",
        interpolation="nearest",
    )
    axes[2].set_title("Spectrogram of the reconstructed audio")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(panel, ax=axes[2], label="Level (dB)")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)
    return destination
