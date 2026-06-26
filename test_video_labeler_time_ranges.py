"""Script-style smoke tests for seconds-range frame selection."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_labeler.time_ranges import frame_indices_for_seconds_range


def main() -> None:
    """Run seconds-range frame selection checks."""
    assert frame_indices_for_seconds_range(1.0, 2.0, fps=30.0, frame_count=100) == list(
        range(30, 61)
    )
    assert frame_indices_for_seconds_range(0.0, 0.1, fps=10.0, frame_count=5) == [0, 1]
    assert frame_indices_for_seconds_range(2.0, 1.0, fps=30.0, frame_count=100) == list(
        range(30, 61)
    )
    assert frame_indices_for_seconds_range(-1.0, 10.0, fps=2.0, frame_count=4) == [0, 1, 2, 3]
    assert frame_indices_for_seconds_range(0.0, 1.0, fps=0.0, frame_count=10) == [0]
    assert frame_indices_for_seconds_range(0.0, 1.0, fps=30.0, frame_count=0) == []

    print("video_labeler time range smoke test passed.")


if __name__ == "__main__":
    main()
