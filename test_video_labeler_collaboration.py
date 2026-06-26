"""Script-style smoke tests for multi-annotator episode collaboration."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_labeler.collaboration import (
    build_even_assignments,
    next_assigned_episode,
    normalize_assignments_payload,
    parse_annotator_names,
)


def main() -> None:
    """Run collaboration-focused pure-Python checks."""
    assert parse_annotator_names("alice, bob\nchen") == ["alice", "bob", "chen"]

    assignments = build_even_assignments(["alice", "bob"], [0, 1, 2, 3, 4])
    assert assignments == {
        "alice": [0, 2, 4],
        "bob": [1, 3],
    }

    payload = {
        "version": 1,
        "assignments": {
            "alice": [0, "2", 4, 4],
            "bob": ["1", 3],
        },
    }
    assert normalize_assignments_payload(payload) == {
        "alice": [0, 2, 4],
        "bob": [1, 3],
    }

    assert normalize_assignments_payload({"alice": [0, 1]}) == {"alice": [0, 1]}
    assert next_assigned_episode([0, 2, 4], current_episode=0, direction=1) == 2
    assert next_assigned_episode([0, 2, 4], current_episode=2, direction=1) == 4
    assert next_assigned_episode([0, 2, 4], current_episode=4, direction=1) == 4
    assert next_assigned_episode([0, 2, 4], current_episode=3, direction=1) == 4
    assert next_assigned_episode([0, 2, 4], current_episode=3, direction=-1) == 2
    assert next_assigned_episode([0, 2, 4], current_episode=0, direction=-1) == 0

    print("video_labeler collaboration smoke test passed.")


if __name__ == "__main__":
    main()
