# autosubs

Automatic subtitle (`.srt`) generator CLI. Point it at video/audio files (or
directories) and it writes a sidecar `.srt` subtitle file per input using a
local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) speech-to-text
model. Runs fully offline after the first model download.

Features:
- Automatic spoken-language detection.
- Translate-to-English mode (`--translate`).
- Batch / recursive directory input.
- Cross-platform: CPU int8 on macOS (Apple Silicon), CUDA on Linux/NVIDIA (untested but should work).

## Try it without installing

With [uv](https://docs.astral.sh/uv/), you can run it once without installing
anything permanently:

```sh
uvx --from autosubs-whisper autosubs VIDEO [VIDEO ...]
```

`uvx` fetches the package into a temporary environment, runs the `autosubs`
command, and leaves nothing behind. (The `--from` flag is needed because the
package is named `autosubs-whisper` (`autosubs` was already taken) while the command is `autosubs`.)

## Install

```sh
uv tool install autosubs-whisper
```

This puts the `autosubs` command on your `PATH` (works on macOS, Linux, and
Windows). `pipx install autosubs-whisper` works too.

For local development from a clone, use `uv sync` and `uv run autosubs`.

## Usage

```sh
autosubs VIDEO [VIDEO ...]
```

The first run downloads the chosen Whisper model from the Hugging Face Hub
(needs internet once); subsequent runs are offline.

Examples:

```sh
# Transcribe one file -> creates video.srt beside it
autosubs talk.mp4

# Recurse a directory, English translation, larger model
autosubs ./season1/ --translate --model large-v3

# Force source language, write into a separate folder, regenerate existing
autosubs clip.mkv --language ja --output-dir ./subs --overwrite
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `small` | Whisper size/name: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, `distil-large-v3`. Use `large-v3` when a CUDA GPU is available. |
| `--translate` | off | Translate speech to English (Whisper `task=translate`). |
| `--language` | auto | Force source language ISO code (`en`, `ja`, ...). Omit to auto-detect. |
| `--device` | `auto` | `auto` (CUDA if available, else CPU), `cpu`, or `cuda`. |
| `--compute-type` | `int8` | CTranslate2 compute type (`int8`, `int8_float16`, `float16`). |
| `--beam-size` | `5` | Beam search width. |
| `--output-dir` | beside input | Directory for `.srt` files. |
| `--overwrite` | off | Regenerate `.srt` files that already exist (default: skip). |

By default, inputs whose target `.srt` already exists are skipped; pass
`--overwrite` to regenerate.

## Notes

- Audio is decoded directly from video containers via PyAV (bundled with
  faster-whisper), so system FFmpeg is not required. If installed, `ffmpeg` is
  used as a fallback for exotic codecs PyAV cannot open.
- On macOS only the CPU backend is available (no Metal GPU support in
  CTranslate2); `int8` keeps it reasonably fast.