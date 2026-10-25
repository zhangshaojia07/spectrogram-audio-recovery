"""Metadata sidecar describing how a spectrogram image was produced.

A spectrogram image alone does not carry the time span, the frequency range,
or the dynamic range that were used to build it.  The encoder therefore writes
a small JSON sidecar next to every image so that decoding is self-contained.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .errors import MetadataError

FORMAT_NAME = "spectrogram-audio"
FORMAT_VERSION = 1
DEFAULT_GENERATOR = "spectrogram-audio"


def metadata_path_for(image_path: str | Path) -> Path:
    """Return the conventional sidecar path for a spectrogram image."""
    return Path(image_path).with_suffix(".json")


@dataclass(frozen=True)
class SpectrogramMetadata:
    """All information required to invert a spectrogram image."""

    sample_rate: int
    n_fft: int
    hop_length: int
    freq_max: float
    min_db: float
    max_db: float
    gain_db: float
    height: int
    width: int
    duration: float
    win_length: int
    generator: str = DEFAULT_GENERATOR
    format: str = FORMAT_NAME
    version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.height < 2:
            raise MetadataError("height must be at least 2 pixels")
        if self.width < 1:
            raise MetadataError("width must be at least 1 pixel")
        if self.duration <= 0:
            raise MetadataError("duration must be positive")
        if self.freq_max <= 0:
            raise MetadataError("freq_max must be positive")
        if self.max_db <= self.min_db:
            raise MetadataError("max_db must be greater than min_db")
        if self.sample_rate <= 0:
            raise MetadataError("sample_rate must be positive")
        if self.n_fft < 2:
            raise MetadataError("n_fft must be at least 2")
        if self.hop_length < 1:
            raise MetadataError("hop_length must be at least 1")
        if self.win_length < 2:
            raise MetadataError("win_length must be at least 2")

    @property
    def output_sample_rate(self) -> int:
        """Sample rate used when reconstructing audio.

        Row ``i`` of the image represents ``i / (height - 1) * freq_max``
        hertz.  The decoder uses ``n_fft = 2 * (height - 1)``, so its bin
        spacing is ``sample_rate / n_fft``.  Rows line up exactly when
        ``sample_rate = 2 * freq_max``, which is also the Nyquist rate for the
        encoded band.
        """
        return int(round(2.0 * self.freq_max))

    @property
    def output_hop_length(self) -> int:
        """Hop length that preserves the encoded duration at the output rate."""
        return max(1, int(round(self.duration * self.output_sample_rate / self.width)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this metadata."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the metadata as UTF-8 JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> Path:
        """Write the metadata to ``path`` and return the resolved path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpectrogramMetadata:
        """Build metadata from a decoded JSON object."""
        if data.get("format") != FORMAT_NAME:
            raise MetadataError(
                f"unsupported metadata format {data.get('format')!r}; expected {FORMAT_NAME!r}"
            )
        if data.get("version") != FORMAT_VERSION:
            raise MetadataError(
                f"unsupported metadata version {data.get('version')!r}; expected {FORMAT_VERSION}"
            )

        known = {field.name for field in fields(cls)}
        missing = sorted(known - data.keys() - {"generator"})
        if missing:
            raise MetadataError("metadata is missing required field(s): " + ", ".join(missing))

        try:
            return cls(**{name: data[name] for name in known if name in data})
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"invalid metadata: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> SpectrogramMetadata:
        """Read metadata from a JSON sidecar."""
        source = Path(path)
        if not source.is_file():
            raise MetadataError(f"metadata file not found: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetadataError(f"cannot read metadata {source}: {exc}") from exc
        if not isinstance(data, dict):
            raise MetadataError(f"metadata file {source} must contain a JSON object")
        return cls.from_dict(data)
