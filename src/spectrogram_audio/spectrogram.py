"""Conversions between STFT spectrograms and 8-bit grayscale images.

The image format is intentionally simple:

* one pixel column per STFT frame (left to right in time),
* one pixel row per frequency bin (low frequencies at the bottom),
* white means "at or below the noise floor", black means "at the peak level",
* a linear mapping between grayscale value and decibels inside the configured
  dynamic range.

Because the mapping is linear and the image is stored as a lossless PNG, the
spectrogram can be recovered up to a known quantization step of
``(max_db - min_db) / 255`` decibels.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from PIL import Image

from .errors import SpectrogramAudioError

UINT8_MAX = 255


def _check_db_range(min_db: float, max_db: float) -> None:
    if not np.isfinite(min_db) or not np.isfinite(max_db):
        raise SpectrogramAudioError("min_db and max_db must be finite")
    if max_db <= min_db:
        raise SpectrogramAudioError("max_db must be greater than min_db")


def stft_db(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 128,
    freq_max: float = 3000.0,
    height: int = 160,
    min_db: float = -80.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """Compute a decibel spectrogram resampled to ``height`` frequency rows.

    Args:
        samples: Mono audio samples.
        sample_rate: Sample rate of ``samples`` in hertz.
        n_fft: FFT size used for the short-time Fourier transform.
        hop_length: Number of samples between successive frames.
        freq_max: Highest frequency to keep, in hertz.  Must not exceed the
            Nyquist frequency of ``sample_rate``.
        height: Number of frequency rows in the returned array.
        min_db: Decibel value mapped to white (minimum level).
        max_db: Decibel value mapped to black (maximum level).

    Returns:
        A ``float32`` array of shape ``(height, frame_count)``.  Row ``0``
        corresponds to 0 Hz and the last row to ``freq_max``.
    """
    y = np.asarray(samples, dtype=np.float32)
    if y.ndim != 1:
        raise SpectrogramAudioError("samples must be a one-dimensional mono signal")
    if y.size == 0:
        raise SpectrogramAudioError("cannot build a spectrogram from empty audio")
    if sample_rate <= 0:
        raise SpectrogramAudioError("sample_rate must be positive")
    if n_fft < 2:
        raise SpectrogramAudioError("n_fft must be at least 2")
    if hop_length < 1:
        raise SpectrogramAudioError("hop_length must be at least 1")
    if height < 2:
        raise SpectrogramAudioError("height must be at least 2 pixels")
    if freq_max <= 0:
        raise SpectrogramAudioError("freq_max must be positive")

    nyquist = sample_rate / 2.0
    if freq_max > nyquist + 1e-6:
        raise SpectrogramAudioError(
            f"freq_max ({freq_max:g} Hz) exceeds the Nyquist frequency "
            f"({nyquist:g} Hz) of the input audio"
        )
    _check_db_range(min_db, max_db)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    with np.errstate(divide="ignore"):
        db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=None)
    db = np.clip(db, min_db, max_db)

    source_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    target_freqs = np.linspace(0.0, freq_max, height)

    resampled = np.empty((height, db.shape[1]), dtype=np.float32)
    for frame in range(db.shape[1]):
        resampled[:, frame] = np.interp(target_freqs, source_freqs, db[:, frame])
    return resampled


def db_to_uint8(
    db_spectrogram: np.ndarray,
    min_db: float = -80.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """Map a decibel spectrogram to a vertically flipped 8-bit image.

    The returned array is ready to be written as a PNG: row ``0`` is the top
    of the image (high frequencies) and the last row is the bottom (0 Hz).
    """
    _check_db_range(min_db, max_db)
    db = np.asarray(db_spectrogram, dtype=np.float64)
    if db.ndim != 2:
        raise SpectrogramAudioError("db_spectrogram must be a two-dimensional array")
    if db.shape[0] < 2 or db.shape[1] < 1:
        raise SpectrogramAudioError("db_spectrogram is too small to render")

    span = float(max_db) - float(min_db)
    gray = (float(max_db) - db) / span * UINT8_MAX
    gray = np.clip(np.rint(gray), 0, UINT8_MAX).astype(np.uint8)
    return np.ascontiguousarray(np.flipud(gray))


def uint8_to_db(
    image: np.ndarray,
    min_db: float = -80.0,
    max_db: float = 0.0,
) -> np.ndarray:
    """Recover a decibel spectrogram from a vertically flipped 8-bit image."""
    _check_db_range(min_db, max_db)
    pixels = np.asarray(image)
    if pixels.ndim != 2:
        raise SpectrogramAudioError("spectrogram image must be two-dimensional")
    if pixels.dtype != np.uint8:
        raise SpectrogramAudioError("spectrogram image must have an 8-bit grayscale layout")

    gray = np.ascontiguousarray(np.flipud(pixels)).astype(np.float32)
    span = float(max_db) - float(min_db)
    return (float(max_db) - gray / UINT8_MAX * span).astype(np.float32)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write an 8-bit grayscale array as a lossless PNG image."""
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise SpectrogramAudioError(
            f"spectrogram images must use the PNG format, got: {destination.suffix or '<none>'}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise SpectrogramAudioError("image must be a two-dimensional uint8 array")
    Image.fromarray(array).save(destination, format="PNG")
    return destination


def load_image(path: str | Path) -> np.ndarray:
    """Read a spectrogram image as an 8-bit grayscale array.

    Color images are converted to luminance, which keeps the decoder usable
    with screenshots and with images produced by older versions of this
    project.  The returned array is in image order (high frequencies on top).
    """
    source = Path(path)
    if not source.is_file():
        raise SpectrogramAudioError(f"spectrogram image not found: {source}")
    try:
        with Image.open(source) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise SpectrogramAudioError(f"cannot read spectrogram image {source}: {exc}") from exc

    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise SpectrogramAudioError(
            f"spectrogram image {source} must be at least 2 pixels high and 1 pixel wide"
        )
    return array
