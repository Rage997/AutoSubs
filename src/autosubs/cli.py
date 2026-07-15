"""Command-line entry point for autosubs."""

import argparse
import os
import sys
from pathlib import Path

from .srt import write_srt
from .transcribe import load_model, transcribe_file

MEDIA_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv",
    ".mpg", ".mpeg", ".ts", ".m4a", ".mp3", ".wav", ".flac", ".ogg",
    ".aac", ".opus",
}


def _expand_inputs(inputs):
    """Expand file/directory paths into a flat list of media files.

    Directories are scanned recursively; only files whose lowercase suffix is
    in ``MEDIA_EXTS`` are kept. Explicit file arguments are always included
    regardless of extension (explicit user intent).
    """
    files = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in MEDIA_EXTS:
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: not found: {raw}", file=sys.stderr)
    return files


def _build_parser():
    p = argparse.ArgumentParser(
        prog="autosubs",
        description="Generate .srt subtitles from video/audio files using local Whisper.",
    )
    p.add_argument("inputs", nargs="+", help="Media file(s) and/or directory(ies) to process.")
    p.add_argument("--model", default="small",
                   help="Whisper model size/name (e.g. tiny, base, small, medium, "
                        "large-v3, large-v3-turbo, distil-large-v3). Default: small. "
                        "large-v3 is recommended when a CUDA GPU is available.")
    p.add_argument("--translate", action="store_true",
                   help="Translate speech to English (task=translate).")
    p.add_argument("--language", default=None,
                   help="Force source language ISO code (e.g. en, ja). Omit to auto-detect.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device. Default: auto (CUDA if available, else CPU).")
    p.add_argument("--compute-type", default="int8",
                   help="CTranslate2 compute type (e.g. int8, int8_float16, float16). "
                        "Default: int8.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam search size. Default: 5.")
    p.add_argument("--output-dir", default=None,
                   help="Directory for .srt files. Default: beside each input.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate .srt files that already exist (default: skip).")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    media_files = _expand_inputs(args.inputs)
    if not media_files:
        print("error: no media files found in the given inputs.", file=sys.stderr)
        return 1

    out_dir = None
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)

    task = "translate" if args.translate else "transcribe"

    print(f"Loading model '{args.model}' (device={args.device}, compute_type={args.compute_type})...")
    model = load_model(args.model, args.device, args.compute_type)

    any_success = False
    any_error = False
    for media in media_files:
        out_path = (out_dir or media.parent) / (media.stem + ".srt")
        if out_path.exists() and not args.overwrite:
            print(f"skip (exists): {out_path}")
            any_success = True
            continue
        try:
            segments, language = transcribe_file(
                model, str(media),
                task=task, language=args.language, beam_size=args.beam_size,
            )
            count = write_srt(segments, str(out_path))
            print(f"{media.name}: detected {language}, {count} segments -> {out_path}")
            if count == 0:
                print(f"warning: no speech segments written for {media}", file=sys.stderr)
            any_success = True
        except Exception as exc:  # noqa: BLE001 - batch resilience: keep going
            print(f"error ({media}): {exc}", file=sys.stderr)
            any_error = True

    if any_success:
        return 0
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
