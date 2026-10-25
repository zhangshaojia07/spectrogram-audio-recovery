"""End-to-end tests for encoding and decoding audio files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from spectrogram_audio.codec import decode_file, encode_file
from spectrogram_audio.errors import SpectrogramAudioError
from spectrogram_audio.metadata import SpectrogramMetadata, metadata_path_for

SAMPLE_RATE = 8000
DURATION = 1.2
FREQ_MAX = 4000.0
HEIGHT = 129


def write_test_audio(path: Path) -> Path:
    time = np.arange(int(SAMPLE_RATE * DURATION), dtype=np.float32) / SAMPLE_RATE
    signal = 0.6 * np.sin(2.0 * np.pi * 440.0 * time)
    signal += 0.3 * np.sin(2.0 * np.pi * 880.0 * time)
    signal *= np.hanning(signal.size).astype(np.float32)
    sf.write(path, signal.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    return path


@pytest.fixture()
def encoded(tmp_path: Path):
    source = write_test_audio(tmp_path / "input.wav")
    return encode_file(
        source,
        tmp_path / "spectrogram.png",
        freq_max=FREQ_MAX,
        height=HEIGHT,
        n_fft=512,
        hop_length=64,
    )


def peak_frequency(audio: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    band = (frequencies >= low) & (frequencies <= high)
    return float(frequencies[band][np.argmax(spectrum[band])])


def test_encode_writes_image_and_matching_metadata(encoded, tmp_path: Path) -> None:
    assert encoded.image_path.is_file()
    assert encoded.metadata_path == metadata_path_for(encoded.image_path)
    assert encoded.metadata_path.is_file()

    metadata = SpectrogramMetadata.load(encoded.metadata_path)
    assert metadata.sample_rate == SAMPLE_RATE
    assert metadata.freq_max == FREQ_MAX
    assert metadata.height == HEIGHT
    assert metadata.width == encoded.image.shape[1]
    assert metadata.duration == pytest.approx(DURATION, abs=1e-6)
    assert metadata.output_sample_rate == int(2 * FREQ_MAX)

    with Image.open(encoded.image_path) as image:
        assert image.mode == "L"
        assert image.size == (metadata.width, HEIGHT)


def test_decode_roundtrip_recovers_frequencies_and_duration(encoded, tmp_path: Path) -> None:
    output = tmp_path / "recovered.wav"
    decoded = decode_file(
        encoded.image_path,
        output,
        metadata=encoded.metadata,
        n_iter=32,
    )

    audio, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == int(2 * FREQ_MAX)
    assert decoded.sample_rate == sample_rate
    assert decoded.duration == pytest.approx(DURATION, abs=0.01)
    assert decoded.peak == pytest.approx(0.95, abs=1e-3)
    assert decoded.n_fft == 2 * (HEIGHT - 1)

    assert abs(peak_frequency(audio, sample_rate, 380.0, 500.0) - 440.0) < 40.0
    assert abs(peak_frequency(audio, sample_rate, 800.0, 960.0) - 880.0) < 40.0


def test_decode_without_metadata_uses_explicit_time_parameters(encoded, tmp_path: Path) -> None:
    output = tmp_path / "explicit.wav"
    decoded = decode_file(
        encoded.image_path,
        output,
        duration=DURATION,
        freq_max=FREQ_MAX,
        min_db=-80.0,
        max_db=0.0,
        n_iter=8,
    )

    assert decoded.metadata is None
    assert output.is_file()
    assert decoded.duration == pytest.approx(DURATION, abs=0.01)


def test_decode_requires_time_parameters_without_metadata(encoded, tmp_path: Path) -> None:
    with pytest.raises(SpectrogramAudioError, match="duration is required"):
        decode_file(encoded.image_path, tmp_path / "out.wav")


def test_decode_rejects_metadata_for_another_image(encoded, tmp_path: Path) -> None:
    wrong = replace(encoded.metadata, width=encoded.metadata.width + 1)

    with pytest.raises(SpectrogramAudioError, match="does not belong"):
        decode_file(encoded.image_path, tmp_path / "out.wav", metadata=wrong)


def test_encode_can_skip_the_metadata_sidecar(tmp_path: Path) -> None:
    source = write_test_audio(tmp_path / "input.wav")
    image_path = tmp_path / "no_sidecar.png"

    result = encode_file(
        source,
        image_path,
        freq_max=FREQ_MAX,
        height=HEIGHT,
        n_fft=512,
        hop_length=64,
        write_metadata=False,
    )

    assert result.metadata_path is None
    assert not metadata_path_for(image_path).exists()


def test_encode_rejects_frequencies_above_nyquist(tmp_path: Path) -> None:
    source = write_test_audio(tmp_path / "input.wav")

    with pytest.raises(SpectrogramAudioError, match="Nyquist"):
        encode_file(source, tmp_path / "bad.png", freq_max=5000.0)


def test_encode_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SpectrogramAudioError, match="audio file not found"):
        encode_file(tmp_path / "missing.wav", tmp_path / "out.png")
