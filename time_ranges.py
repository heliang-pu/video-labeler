"""Helpers for converting seconds-based labels into frame indices."""

from __future__ import annotations


def frame_indices_for_seconds_range(
    start_seconds: float,
    end_seconds: float,
    *,
    fps: float,
    frame_count: int,
) -> list[int]:
    """Return every frame index covered by one seconds interval."""
    if frame_count <= 0:
        return []

    normalized_start = max(0.0, float(start_seconds))
    normalized_end = max(0.0, float(end_seconds))
    if normalized_start > normalized_end:
        normalized_start, normalized_end = normalized_end, normalized_start

    if fps <= 0:
        return [0]

    start_frame = int(round(normalized_start * fps))
    end_frame = int(round(normalized_end * fps))
    start_frame = max(0, min(start_frame, frame_count - 1))
    end_frame = max(0, min(end_frame, frame_count - 1))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return list(range(start_frame, end_frame + 1))
