"""Transcription core — the only module that touches faster-whisper.

Centralizes ``WhisperModel`` construction, the ``.transcribe(...)`` call, and
the PyAV -> system-ffmpeg decode fallback. Returns faster-whisper segment
objects unchanged (see ``srt.py`` for the segment contract).
"""

import os
import shutil
import subprocess
import tempfile

from faster_whisper import WhisperModel


def load_model(model: str, device: str, compute_type: str) -> WhisperModel:
    """Construct a ``WhisperModel``.

    ``device`` is ``"auto" | "cpu" | "cuda"`` (CTranslate2 resolves ``"auto"``
    to CUDA when available, else CPU). ``compute_type`` is passed through
    (e.g. ``"int8"``, ``"int8_float16"``, ``"float16"``).
    """
    return WhisperModel(model, device=device, compute_type=compute_type)


def _decode_and_transcribe(model, audio_path, *, task, language, beam_size):
    """Run ``model.transcribe`` on ``audio_path``, materializing the segments.

    faster-whisper decodes lazily, so decode errors surface only when the
    generator is drained. On any decode failure, fall back to extracting a
    16 kHz mono WAV via system ffmpeg (if present) and retry; otherwise raise a
    clear ``RuntimeError``.

    ``word_timestamps=True`` is required for correct alignment: without it
    faster-whisper only emits coarse segment timestamps that snap to VAD-chunk
    and 30s-window boundaries, so a subtitle's start/end can drift 10+ seconds
    across surrounding silence. Word-level DTW alignment pins ``segment.start``
    to the first word and ``segment.end`` to the last.
    """
    try:
        segments, info = model.transcribe(
            audio_path,
            task=task,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=True,
        )
        return list(segments), info
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"Could not decode audio from {audio_path}; "
                "install ffmpeg for broader format support."
            )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav = tmp.name
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-i", audio_path,
             "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav, "-y"],
            check=True,
            capture_output=True,
        )
        segments, info = model.transcribe(
            tmp_wav,
            task=task,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=True,
        )
        return list(segments), info
    finally:
        os.unlink(tmp_wav)


def transcribe_file(model, audio_path: str, *, task: str, language, beam_size: int):
    """Transcribe (or translate) ``audio_path``.

    ``task`` is ``"transcribe"`` or ``"translate"`` (Whisper always translates
    to English). ``language=None`` auto-detects the source language.
    Returns ``(segments, detected_language)``.
    """
    segments, info = _decode_and_transcribe(
        model, audio_path, task=task, language=language, beam_size=beam_size
    )
    return segments, info.language
