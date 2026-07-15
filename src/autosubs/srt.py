"""SRT subtitle formatter.

Segment contract: accepts any iterable of objects exposing float ``.start``,
float ``.end``, str ``.text`` (faster-whisper segment objects satisfy this
directly; no wrapper dataclass is used).
"""


def _format_timestamp(seconds: float) -> str:
    """Format a float second offset as an SRT timestamp ``HH:MM:SS,mmm``.

    SRT uses a comma (not a period) as the millisecond separator.
    """
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    h, ms_total = divmod(ms_total, 3600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, out_path: str) -> int:
    """Write ``segments`` to ``out_path`` as UTF-8 SRT.

    Each block is ``<index>\\n<start> --> <end>\\n<text>\\n\\n`` with a 1-based
    index. Segments whose stripped text is empty are skipped (VAD/hallucination
    guard). The file is always created, even when zero blocks are written.
    Returns the number of blocks written.
    """
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            count += 1
            f.write(f"{count}\n")
            f.write(f"{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n")
            f.write(f"{text}\n\n")
    return count
