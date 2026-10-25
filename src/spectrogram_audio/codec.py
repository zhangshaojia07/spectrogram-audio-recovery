"""High-level encode and decode operations built on the image primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import spectrogram
from .errors import SpectrogramAudioError
from .metadata import SpectrogramMetadata, metadata_path_for

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 128
DEFAULT_FREQ_MAX = 3000.0
DEFAULT_HEIGHT = 160
DEFAULT_MIN_DB = -80.0
DEFAULT_MAX_DB = 0.0
DEFAULT_GAIN_DB = 0.0
DEFAULT_WIN_LENGTH = 128
DEFAULT_N_ITER = 32
DEFAULT_PEAK = 0.95


@dataclass(frozen=True)
class EncodeResult:
    """Outcome of encoding an audio file as a spectrogram image."""

    image_path: Path
    metadata_path: Path | None
    metadata: SpectrogramMetadata
    image: np.ndarray

    @property
    def duration(self) -> float:
        return self.metadata.duration


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of reconstructing audio from a spectrogram image."""

    audio_path: Path
    audio: np.ndarray
    sample_rate: int
    metadata: SpectrogramMetadata | None
    n_fft: int
    hop_length: int
    n_iter: int

    @property
    def duration(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.audio))) if self.audio.size else 0.0

    @property
    def rms(self) -> float:
        if not self.audio.size:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.audio, dtype=np.float64))))


def _validate_db_params(min_db: float, max_db: float, gain_db: float) -> None:
    if not np.isfinite(min_db) or not np.isfinite(max_db) or not np.isfinite(gain_db):
        raise SpectrogramAudioError("min_db, max_db, and gain_db must be finite")
    if max_db <= min_db:
        raise SpectrogramAudioError("max_db must be greater than min_db")


def encode_file(
    audio_path: str | Path,
    image_path: str | Path,
    *,
    freq_max: float = DEFAULT_FREQ_MAX,
    height: int = DEFAULT_HEIGHT,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    min_db: float = DEFAULT_MIN_DB,
    max_db: float = DEFAULT_MAX_DB,
    gain_db: float = DEFAULT_GAIN_DB,
    win_length: int = DEFAULT_WIN_LENGTH,
    write_metadata: bool = True,
    metadata_path: str | Path | None = None,
) -> EncodeResult:
    """Encode an audio file as a grayscale spectrogram image.

    The input is mixed down to mono and analyzed at its native sample rate.
    A JSON sidecar containing the decoding parameters is written next to the
    image unless ``write_metadata`` is false.

    Args:
        audio_path: Input audio file readable by librosa/soundfile.
        image_path: Destination path for the PNG spectrogram.
        freq_max: Highest frequency to encode, in hertz.
        height: Image height in pixels (frequency resolution).
        n_fft: FFT size used for the analysis.
        hop_length: Analysis hop size in input samples (time resolution).
        min_db: Level mapped to white.
        max_db: Level mapped to black.
        gain_db: Amplitude offset stored for the decoder.
        win_length: Window length recommended to the decoder.
        write_metadata: Whether to write the JSON sidecar.
        metadata_path: Optional custom sidecar path.

    Returns:
        An :class:`EncodeResult` with the written paths and image array.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise SpectrogramAudioError(f"audio file not found: {source}")
    _validate_db_params(min_db, max_db, gain_db)
    if height < 2:
        raise SpectrogramAudioError("height must be at least 2 pixels")
    if win_length < 2:
        raise SpectrogramAudioError("win_length must be at least 2")

    try:
        samples, sample_rate = librosa.load(str(source), sr=None, mono=True)
    except Exception as exc:  # librosa raises several backend-specific errors
        raise SpectrogramAudioError(f"cannot read audio file {source}: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise SpectrogramAudioError(f"audio file {source} contains no samples")

    db_spectrogram = spectrogram.stft_db(
        samples,
        sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_max=freq_max,
        height=height,
        min_db=min_db,
        max_db=max_db,
    )
    image = spectrogram.db_to_uint8(db_spectrogram, min_db, max_db)
    destination = spectrogram.save_image(image, image_path)

    metadata = SpectrogramMetadata(
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        freq_max=float(freq_max),
        min_db=float(min_db),
        max_db=float(max_db),
        gain_db=float(gain_db),
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        duration=float(samples.size / sample_rate),
        win_length=int(win_length),
    )

    sidecar: Path | None = None
    if write_metadata:
        sidecar = Path(metadata_path) if metadata_path else metadata_path_for(destination)
        metadata.save(sidecar)

    return EncodeResult(
        image_path=destination,
        metadata_path=sidecar,
        metadata=metadata,
        image=image,
    )


def decode_spectrogram(
    db_spectrogram: np.ndarray,
    *,
    duration: float,
    freq_max: float,
    gain_db: float = DEFAULT_GAIN_DB,
    n_iter: int = DEFAULT_N_ITER,
    win_length: int = DEFAULT_WIN_LENGTH,
    normalize: bool = True,
    random_state: int = 0,
) -> tuple[np.ndarray, int, int, int]:
    """Reconstruct a waveform from a decibel spectrogram with Griffin-Lim.

    Args:
        db_spectrogram: Array of shape ``(height, width)`` in decibels, with
            row ``0`` at 0 Hz.
        duration: Target duration of the reconstructed audio in seconds.
        freq_max: Highest encoded frequency in hertz.
        gain_db: Global amplitude offset applied to the linear spectrogram.
        n_iter: Number of Griffin-Lim iterations.
        win_length: Analysis window length used during phase retrieval.
        normalize: Scale the result to a fixed peak level for playback.
        random_state: Seed for the initial phase estimate.

    Returns:
        ``(audio, sample_rate, n_fft, hop_length)``.
    """
    db = np.asarray(db_spectrogram, dtype=np.float32)
    if db.ndim != 2 or db.shape[0] < 2 or db.shape[1] < 1:
        raise SpectrogramAudioError("spectrogram must have shape (height >= 2, width >= 1)")
    if duration <= 0:
        raise SpectrogramAudioError("duration must be positive")
    if freq_max <= 0:
        raise SpectrogramAudioError("freq_max must be positive")
    if n_iter < 1:
        raise SpectrogramAudioError("n_iter must be at least 1")
    if not np.isfinite(gain_db):
        raise SpectrogramAudioError("gain_db must be finite")

    height, width = db.shape
    n_fft = 2 * (height - 1)
    sample_rate = int(round(2.0 * freq_max))
    hop_length = max(1, int(round(duration * sample_rate / width)))
    target_length = max(1, int(round(duration * sample_rate)))
    window = min(int(win_length), n_fft)
    if window < 2:
        raise SpectrogramAudioError("win_length must be at least 2 after clamping to n_fft")

    # The encoded image stores dB relative to the loudest bin, so the absolute
    # scale is arbitrary.  Decibels are amplitudes (20 log10), hence the divisor
    # of 20; peak normalization then restores a usable playback level.
    amplitude = np.power(10.0, (db + float(gain_db)) / 20.0).astype(np.float32)
    audio = librosa.griffinlim(
        amplitude,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window,
        n_iter=int(n_iter),
        rng=random_state,
        length=target_length,
    )
    audio = np.asarray(audio, dtype=np.float32)

    if normalize and audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio * (DEFAULT_PEAK / peak)

    return audio, sample_rate, n_fft, hop_length


def decode_file(
    image_path: str | Path,
    output_path: str | Path,
    *,
    metadata: SpectrogramMetadata | None = None,
    duration: float | None = None,
    freq_max: float | None = None,
    min_db: float | None = None,
    max_db: float | None = None,
    gain_db: float | None = None,
    win_length: int | None = None,
    n_iter: int = DEFAULT_N_ITER,
    normalize: bool = True,
    random_state: int = 0,
    plot_path: str | Path | None = None,
) -> DecodeResult:
    """Reconstruct a WAV file from a spectrogram image.

    Values supplied by ``metadata`` are used for every parameter that is not
    passed explicitly.  Without metadata, ``duration`` and ``freq_max`` are
    mandatory because they cannot be inferred from the pixels alone.

    Args:
        image_path: Spectrogram image (PNG or any format readable by Pillow).
        output_path: Destination WAV path.
        metadata: Optional decoding parameters, usually read from a sidecar.
        duration: Duration of the original audio in seconds.
        freq_max: Highest encoded frequency in hertz.
        min_db: Level encoded as white.
        max_db: Level encoded as black.
        gain_db: Global amplitude offset applied before phase retrieval.
        win_length: Griffin-Lim window length.
        n_iter: Number of Griffin-Lim iterations.
        normalize: Peak-normalize the reconstructed audio.
        random_state: Seed for the initial phase estimate.
        plot_path: Optional path for a diagnostic figure.

    Returns:
        A :class:`DecodeResult` describing the reconstruction.
    """
    image = spectrogram.load_image(image_path)
    height, width = image.shape

    if metadata is not None:
        if (metadata.height, metadata.width) != (height, width):
            raise SpectrogramAudioError(
                f"image is {width}x{height} px but metadata describes "
                f"{metadata.width}x{metadata.height} px; the sidecar does not belong "
                f"to this image"
            )
        duration = metadata.duration if duration is None else duration
        freq_max = metadata.freq_max if freq_max is None else freq_max
        min_db = metadata.min_db if min_db is None else min_db
        max_db = metadata.max_db if max_db is None else max_db
        gain_db = metadata.gain_db if gain_db is None else gain_db
        win_length = metadata.win_length if win_length is None else win_length

    if duration is None:
        raise SpectrogramAudioError(
            "duration is required when no metadata sidecar is available; "
            "pass duration or provide metadata"
        )
    if freq_max is None:
        raise SpectrogramAudioError(
            "freq_max is required when no metadata sidecar is available; "
            "pass freq_max or provide metadata"
        )
    min_db = DEFAULT_MIN_DB if min_db is None else min_db
    max_db = DEFAULT_MAX_DB if max_db is None else max_db
    gain_db = DEFAULT_GAIN_DB if gain_db is None else gain_db
    win_length = DEFAULT_WIN_LENGTH if win_length is None else win_length

    db = spectrogram.uint8_to_db(image, min_db, max_db)
    audio, sample_rate, n_fft, hop_length = decode_spectrogram(
        db,
        duration=duration,
        freq_max=freq_max,
        gain_db=gain_db,
        n_iter=n_iter,
        win_length=win_length,
        normalize=normalize,
        random_state=random_state,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="PCM_16")

    if plot_path is not None:
        from .plotting import plot_diagnostics

        plot_diagnostics(
            db,
            audio,
            sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            freq_max=float(freq_max),
            duration=duration,
            output_path=plot_path,
        )

    return DecodeResult(
        audio_path=destination,
        audio=audio,
        sample_rate=sample_rate,
        metadata=metadata,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=int(n_iter),
    )
