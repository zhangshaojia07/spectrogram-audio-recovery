"""Encode audio as a spectrogram image and recover it with Griffin-Lim.

The package is intentionally split into small layers:

``spectrogram``
    Audio to decibel spectrogram to 8-bit grayscale image and back.
``codec``
    File-oriented encode and decode operations.
``metadata``
    JSON sidecar that makes a spectrogram image self-describing.
``plotting``
    Optional diagnostic figures (requires the ``plot`` extra).
``cli``
    The ``spectrogram-audio`` command-line interface.
"""

from importlib.metadata import PackageNotFoundError, version

from .codec import (
    DecodeResult,
    EncodeResult,
    decode_file,
    decode_spectrogram,
    encode_file,
)
from .errors import MetadataError, SpectrogramAudioError
from .metadata import SpectrogramMetadata, metadata_path_for
from .spectrogram import db_to_uint8, load_image, save_image, stft_db, uint8_to_db

try:
    __version__ = version("spectrogram-audio-recovery")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = [
    "DecodeResult",
    "EncodeResult",
    "MetadataError",
    "SpectrogramAudioError",
    "SpectrogramMetadata",
    "__version__",
    "db_to_uint8",
    "decode_file",
    "decode_spectrogram",
    "encode_file",
    "load_image",
    "metadata_path_for",
    "save_image",
    "stft_db",
    "uint8_to_db",
]
