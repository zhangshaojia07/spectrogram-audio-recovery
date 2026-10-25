"""Exception types used across the package."""

from __future__ import annotations


class SpectrogramAudioError(Exception):
    """Base class for all errors raised by this package."""


class MetadataError(SpectrogramAudioError, ValueError):
    """Raised when a metadata sidecar is missing, malformed, or unsupported."""
