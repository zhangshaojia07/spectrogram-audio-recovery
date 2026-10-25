"""Tests for the JSON metadata sidecar."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from spectrogram_audio.errors import MetadataError
from spectrogram_audio.metadata import (
    FORMAT_NAME,
    FORMAT_VERSION,
    SpectrogramMetadata,
    metadata_path_for,
)


def make_metadata(**overrides) -> SpectrogramMetadata:
    values = {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 64,
        "freq_max": 8000.0,
        "min_db": -80.0,
        "max_db": 0.0,
        "gain_db": 0.0,
        "height": 257,
        "width": 626,
        "duration": 2.5,
        "win_length": 128,
    }
    values.update(overrides)
    return SpectrogramMetadata(**values)


def test_metadata_save_and_load_roundtrip(tmp_path: Path) -> None:
    metadata = make_metadata()
    path = tmp_path / "spectrogram.json"

    assert metadata.save(path) == path
    loaded = SpectrogramMetadata.load(path)

    assert loaded == metadata
    assert loaded.format == FORMAT_NAME
    assert loaded.version == FORMAT_VERSION


def test_metadata_derives_output_parameters() -> None:
    metadata = make_metadata()

    assert metadata.output_sample_rate == 16000
    assert metadata.output_hop_length == 64


def test_metadata_path_matches_the_image(tmp_path: Path) -> None:
    assert metadata_path_for(tmp_path / "spectrogram.png") == tmp_path / "spectrogram.json"


def test_metadata_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = make_metadata().to_dict()
    data["format"] = "something-else"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MetadataError, match="unsupported metadata format"):
        SpectrogramMetadata.load(path)


def test_metadata_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = make_metadata().to_dict()
    data["version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MetadataError, match="unsupported metadata version"):
        SpectrogramMetadata.load(path)


def test_metadata_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = make_metadata().to_dict()
    del data["duration"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MetadataError, match="missing required field"):
        SpectrogramMetadata.load(path)


def test_metadata_rejects_invalid_ranges() -> None:
    with pytest.raises(MetadataError, match="greater than min_db"):
        make_metadata(min_db=0.0, max_db=-80.0)
    with pytest.raises(MetadataError, match="positive"):
        make_metadata(duration=0.0)


def test_metadata_rejects_wrong_python_types(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    data = make_metadata().to_dict()
    data["height"] = "tall"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MetadataError, match="invalid metadata"):
        SpectrogramMetadata.load(path)


def test_replace_keeps_metadata_usable() -> None:
    metadata = make_metadata()
    changed = replace(metadata, width=627)

    assert changed.width == 627
    assert changed.height == metadata.height
