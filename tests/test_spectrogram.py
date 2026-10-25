"""Tests for the low-level spectrogram and image conversions."""

from __future__ import annotations

import numpy as np
import pytest

from spectrogram_audio.errors import SpectrogramAudioError
from spectrogram_audio.spectrogram import db_to_uint8, save_image, stft_db, uint8_to_db


def test_db_image_roundtrip_is_within_one_quantization_step() -> None:
    rng = np.random.default_rng(0)
    db = rng.uniform(-80.0, 0.0, size=(160, 64)).astype(np.float32)

    image = db_to_uint8(db, min_db=-80.0, max_db=0.0)
    restored = uint8_to_db(image, min_db=-80.0, max_db=0.0)

    step = 80.0 / 255.0
    assert image.shape == db.shape
    assert image.dtype == np.uint8
    assert np.max(np.abs(restored - db)) <= step / 2.0 + 1e-5


def test_strong_frequencies_are_dark_and_low_frequencies_are_at_the_bottom() -> None:
    db = np.full((128, 16), -80.0, dtype=np.float32)
    db[64, :] = 0.0

    image = db_to_uint8(db, min_db=-80.0, max_db=0.0)

    # Row 64 of the dB matrix is the middle of the frequency axis.  Since the
    # image is flipped for storage, it must appear at row 63 from the top.
    assert int(np.argmin(image[:, 0])) == 63
    assert image[63, 0] == 0
    assert image[0, 0] == 255
    assert image[-1, 0] == 255


def test_stft_db_has_requested_shape_and_dynamic_range() -> None:
    sample_rate = 8000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)

    db = stft_db(
        samples,
        sample_rate,
        n_fft=512,
        hop_length=64,
        freq_max=4000.0,
        height=129,
        min_db=-80.0,
        max_db=0.0,
    )

    assert db.shape == (129, 1 + samples.size // 64)
    assert db.min() >= -80.0
    assert db.max() <= 0.0
    assert db[:, 10].max() == pytest.approx(0.0, abs=1e-4)


def test_stft_db_rejects_frequencies_above_nyquist() -> None:
    samples = np.zeros(100, dtype=np.float32)

    with pytest.raises(SpectrogramAudioError, match="Nyquist"):
        stft_db(samples, 8000, freq_max=5000.0)


def test_stft_db_rejects_empty_audio() -> None:
    with pytest.raises(SpectrogramAudioError, match="empty"):
        stft_db(np.array([], dtype=np.float32), 8000)


def test_db_to_uint8_rejects_invalid_range() -> None:
    db = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(SpectrogramAudioError, match="greater than min_db"):
        db_to_uint8(db, min_db=0.0, max_db=-80.0)


def test_save_image_requires_png(tmp_path) -> None:
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(SpectrogramAudioError, match="PNG"):
        save_image(image, tmp_path / "spectrogram.jpg")
