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

    Timestamp and transcription accuracy relies on:
      - word_timestamps=True: pins segment.start/.end to first/last word via DTW
        instead of coarse 30s-window boundaries (can drift 10+ seconds otherwise).
      - condition_on_previous_text=False: prevents decoder from feeding prior text
        as a prompt, reducing repetition and phantom transcriptions.
      - hallucination_silence_threshold=2.0: drops anomalous segments surrounded
        by >2s silence, preventing hallucinated words from appearing seconds before
        the actual line.
      - vad_filter=False: Voice Activity Detection is disabled for maximum accuracy.
        All audio is processed by Whisper, ensuring quiet dialogue, overlapping speech,
        and speech masked by music/sound effects is captured. This is SLOW: expect
        approximately 1x realtime or slower (a 42-minute episode may take 40-60 minutes).
        For faster processing (~10x realtime) at the cost of potentially missing quiet
        dialogue, you can enable VAD by setting vad_filter=True and vad_parameters with
        appropriate thresholds.
    """
    try:
        segments, info = model.transcribe(
            audio_path,
            task=task,
            language=language,
            beam_size=beam_size,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
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
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
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
