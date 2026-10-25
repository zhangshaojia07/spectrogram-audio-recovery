"""Command-line interface for the spectrogram audio codec."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .codec import (
    DEFAULT_FREQ_MAX,
    DEFAULT_GAIN_DB,
    DEFAULT_HEIGHT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_MAX_DB,
    DEFAULT_MIN_DB,
    DEFAULT_N_FFT,
    DEFAULT_N_ITER,
    DEFAULT_WIN_LENGTH,
    DecodeResult,
    EncodeResult,
    decode_file,
    encode_file,
)
from .errors import SpectrogramAudioError
from .metadata import SpectrogramMetadata, metadata_path_for


def _add_encode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--freq-max",
        type=float,
        default=DEFAULT_FREQ_MAX,
        metavar="HZ",
        help="highest frequency to encode",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        metavar="PX",
        help="image height in pixels (frequency resolution)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
        help="FFT size used for the analysis",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="hop size in input samples (time resolution)",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help="level mapped to white",
    )
    parser.add_argument(
        "--max-db",
        type=float,
        default=DEFAULT_MAX_DB,
        help="level mapped to black",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=DEFAULT_GAIN_DB,
        help="amplitude offset stored for the decoder",
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=DEFAULT_WIN_LENGTH,
        help="window length recommended to the decoder",
    )


def _add_metadata_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="do not write a JSON metadata sidecar",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="custom path for the JSON sidecar",
    )


def _add_decode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        metavar="PATH",
        help="metadata sidecar to use (default: <image>.json when present)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="ignore any metadata sidecar",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="duration of the original audio (required without metadata)",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        metavar="HZ",
        help="highest encoded frequency (required without metadata)",
    )
    parser.add_argument("--min-db", type=float, default=None, help="level encoded as white")
    parser.add_argument("--max-db", type=float, default=None, help="level encoded as black")
    parser.add_argument("--gain-db", type=float, default=None, help="amplitude offset in dB")
    parser.add_argument(
        "--win-length",
        type=int,
        default=None,
        help="Griffin-Lim window length",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help="number of Griffin-Lim iterations",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="keep the raw reconstruction level instead of peak normalization",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="write a diagnostic figure (requires the plot extra)",
    )


def _add_decode_short_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help="number of Griffin-Lim iterations",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="keep the raw reconstruction level instead of peak normalization",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="write a diagnostic figure (requires the plot extra)",
    )


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults only when they carry useful information."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is False:
            return action.help or ""
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="spectrogram-audio",
        description=(
            "Encode audio as a grayscale spectrogram image and reconstruct a "
            "waveform from such an image with Griffin-Lim phase retrieval."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser(
        "encode",
        help="encode an audio file as a spectrogram image",
        description="Analyze an audio file and write a grayscale PNG plus metadata.",
        formatter_class=_HelpFormatter,
    )
    encode.add_argument("input", type=Path, help="input audio file")
    encode.add_argument("-o", "--output", type=Path, required=True, help="output PNG image")
    _add_encode_options(encode)
    _add_metadata_options(encode)
    encode.set_defaults(handler=_run_encode)

    decode = subparsers.add_parser(
        "decode",
        help="reconstruct audio from a spectrogram image",
        description="Read a spectrogram image and synthesize a WAV file with Griffin-Lim.",
        formatter_class=_HelpFormatter,
    )
    decode.add_argument("input", type=Path, help="input spectrogram image")
    decode.add_argument("-o", "--output", type=Path, required=True, help="output WAV file")
    _add_decode_options(decode)
    decode.set_defaults(handler=_run_decode)

    roundtrip = subparsers.add_parser(
        "roundtrip",
        help="encode and decode in a single command",
        description="Encode an audio file, then reconstruct it from the generated image.",
        formatter_class=_HelpFormatter,
    )
    roundtrip.add_argument("input", type=Path, help="input audio file")
    roundtrip.add_argument("-o", "--output", type=Path, required=True, help="output WAV file")
    roundtrip.add_argument(
        "--image-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="output PNG image (default: derived from the output WAV path)",
    )
    _add_encode_options(roundtrip)
    _add_metadata_options(roundtrip)
    _add_decode_short_options(roundtrip)
    roundtrip.set_defaults(handler=_run_roundtrip)

    return parser


def _load_metadata(
    image_path: Path,
    explicit: Path | None,
    disabled: bool,
) -> tuple[SpectrogramMetadata | None, Path | None]:
    if disabled:
        return None, None
    if explicit is not None:
        if not explicit.is_file():
            raise SpectrogramAudioError(f"metadata file not found: {explicit}")
        return SpectrogramMetadata.load(explicit), explicit
    candidate = metadata_path_for(image_path)
    if candidate.is_file():
        return SpectrogramMetadata.load(candidate), candidate
    return None, None


def _print_encode_summary(result: EncodeResult) -> None:
    meta = result.metadata
    print(f"Wrote spectrogram image : {result.image_path}")
    print(f"  image size      : {meta.width} x {meta.height} px")
    print(f"  source          : {meta.sample_rate} Hz, {meta.duration:.3f} s")
    print(f"  frequency range : 0 - {meta.freq_max:g} Hz")
    print(f"  dynamic range   : {meta.min_db:g} - {meta.max_db:g} dB")
    if result.metadata_path is not None:
        print(f"  metadata        : {result.metadata_path}")


def _print_decode_summary(result: DecodeResult, metadata_path: Path | None) -> None:
    print(f"Wrote audio             : {result.audio_path}")
    print(f"  duration        : {result.duration:.3f} s")
    print(f"  sample rate     : {result.sample_rate} Hz")
    print(f"  STFT            : n_fft={result.n_fft}, hop={result.hop_length}")
    print(f"  Griffin-Lim     : {result.n_iter} iterations")
    print(f"  peak / RMS      : {result.peak:.3f} / {result.rms:.3f}")
    if metadata_path is not None:
        print(f"  metadata        : {metadata_path}")


def _run_encode(args: argparse.Namespace) -> int:
    result = encode_file(
        args.input,
        args.output,
        freq_max=args.freq_max,
        height=args.height,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        min_db=args.min_db,
        max_db=args.max_db,
        gain_db=args.gain_db,
        win_length=args.win_length,
        write_metadata=not args.no_metadata,
        metadata_path=args.metadata_output,
    )
    _print_encode_summary(result)
    return 0


def _run_decode(args: argparse.Namespace) -> int:
    metadata, metadata_path = _load_metadata(args.input, args.metadata, args.no_metadata)
    result = decode_file(
        args.input,
        args.output,
        metadata=metadata,
        duration=args.duration,
        freq_max=args.freq_max,
        min_db=args.min_db,
        max_db=args.max_db,
        gain_db=args.gain_db,
        win_length=args.win_length,
        n_iter=args.n_iter,
        normalize=not args.no_normalize,
        plot_path=args.plot,
    )
    _print_decode_summary(result, metadata_path)
    return 0


def _run_roundtrip(args: argparse.Namespace) -> int:
    image_path = args.image_out or args.output.with_suffix(".png")
    encoded = encode_file(
        args.input,
        image_path,
        freq_max=args.freq_max,
        height=args.height,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        min_db=args.min_db,
        max_db=args.max_db,
        gain_db=args.gain_db,
        win_length=args.win_length,
        write_metadata=not args.no_metadata,
        metadata_path=args.metadata_output,
    )
    print("[1/2] Encode")
    _print_encode_summary(encoded)

    decoded = decode_file(
        image_path,
        args.output,
        metadata=encoded.metadata,
        n_iter=args.n_iter,
        normalize=not args.no_normalize,
        plot_path=args.plot,
    )
    print("[2/2] Decode")
    _print_decode_summary(decoded, encoded.metadata_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except SpectrogramAudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
