"""Generate a small synthetic clip used by the README examples.

The signal is a sequence of short harmonic sweeps.  It contains no speech or
other personal data, so it is safe to publish alongside the project, and its
spectrogram clearly shows the time-frequency structure that the codec stores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 2.5
SEGMENTS = (
    (0.00, 0.55, 180.0, 320.0),
    (0.65, 1.20, 320.0, 520.0),
    (1.30, 1.85, 240.0, 180.0),
    (1.95, 2.45, 400.0, 700.0),
)


def build_signal() -> np.ndarray:
    """Return a deterministic harmonic-sweep test signal."""
    time = np.arange(int(SAMPLE_RATE * DURATION), dtype=np.float64) / SAMPLE_RATE
    signal = np.zeros_like(time)

    for start, end, start_hz, end_hz in SEGMENTS:
        mask = (time >= start) & (time < end)
        local = (time[mask] - start) / (end - start)
        frequency = start_hz + (end_hz - start_hz) * local
        phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE

        # A sub-harmonic and seven partials create a speech-like stack of
        # horizontal bands in the spectrogram.
        tone = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 9))
        tone += 0.6 * np.sin(0.5 * phase)
        envelope = np.sin(np.pi * local) ** 1.5
        signal[mask] += 0.25 * envelope * tone

    # Add a quiet high-frequency sweep so the upper part of the image is not
    # empty, which makes the demo more informative.
    chirp_phase = 2.0 * np.pi * np.cumsum(2500.0 + 1200.0 * time) / SAMPLE_RATE
    chirp_envelope = np.clip(np.sin(np.pi * time / DURATION), 0.0, None) ** 3
    signal += 0.02 * chirp_envelope * np.sin(chirp_phase)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal *= 0.8 / peak
    return signal.astype(np.float32)


def main() -> None:
    destination = Path(__file__).with_name("demo.wav")
    sf.write(destination, build_signal(), SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {destination} ({DURATION:g} s at {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
