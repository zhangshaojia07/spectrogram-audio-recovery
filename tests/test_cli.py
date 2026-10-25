"""Tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from spectrogram_audio.cli import main


def make_audio(path: Path, sample_rate: int = 8000, duration: float = 1.2) -> Path:
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    signal = 0.7 * np.sin(2.0 * np.pi * 440.0 * time)
    signal *= np.hanning(signal.size).astype(np.float32)
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def encode_arguments(source: Path, image: Path) -> list[str]:
    return [
        "encode",
        str(source),
        "-o",
        str(image),
        "--freq-max",
        "4000",
        "--height",
        "65",
        "--n-fft",
        "256",
        "--hop-length",
        "32",
    ]


def test_cli_roundtrip_writes_all_outputs(tmp_path: Path, capsys) -> None:
    source = make_audio(tmp_path / "input.wav")
    image = tmp_path / "spectrogram.png"
    output = tmp_path / "recovered.wav"

    code = main(
        [
            "roundtrip",
            str(source),
            "-o",
            str(output),
            "--image-out",
            str(image),
            "--freq-max",
            "4000",
            "--height",
            "65",
            "--n-fft",
            "256",
            "--hop-length",
            "32",
            "--n-iter",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert image.is_file()
    assert output.is_file()
    assert image.with_suffix(".json").is_file()
    assert "[1/2] Encode" in captured.out
    assert "[2/2] Decode" in captured.out


def test_cli_decode_uses_the_metadata_sidecar(tmp_path: Path, capsys) -> None:
    source = make_audio(tmp_path / "input.wav")
    image = tmp_path / "spectrogram.png"
    output = tmp_path / "recovered.wav"

    assert main(encode_arguments(source, image)) == 0
    capsys.readouterr()

    assert main(["decode", str(image), "-o", str(output), "--n-iter", "4"]) == 0
    captured = capsys.readouterr()
    assert output.is_file()
    assert "metadata" in captured.out
    assert "sample rate" in captured.out


def test_cli_reports_missing_time_parameters(tmp_path: Path, capsys) -> None:
    source = make_audio(tmp_path / "input.wav")
    image = tmp_path / "spectrogram.png"
    output = tmp_path / "recovered.wav"

    assert main([*encode_arguments(source, image), "--no-metadata"]) == 0
    capsys.readouterr()

    assert main(["decode", str(image), "-o", str(output)]) == 1
    captured = capsys.readouterr()
    assert "duration is required" in captured.err
    assert not output.exists()


def test_cli_encode_can_omit_metadata(tmp_path: Path) -> None:
    source = make_audio(tmp_path / "input.wav")
    image = tmp_path / "spectrogram.png"

    assert main([*encode_arguments(source, image), "--no-metadata"]) == 0
    assert image.is_file()
    assert not image.with_suffix(".json").exists()


def test_cli_version_exits_successfully(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert "spectrogram-audio" in capsys.readouterr().out
