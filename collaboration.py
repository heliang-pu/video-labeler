"""Helpers for assigning LeRobot episodes to multiple annotators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ASSIGNMENTS_FILE_NAME = "label_assignments.json"


def parse_annotator_names(raw_text: str) -> list[str]:
    """Parse comma/newline separated annotator names into a stable unique list."""
    names: list[str] = []
    seen: set[str] = set()
    for chunk in raw_text.replace("\n", ",").split(","):
        name = chunk.strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def build_even_assignments(
    annotator_names: list[str],
    episode_indices: list[int],
) -> dict[str, list[int]]:
    """Assign episodes round-robin across annotators."""
    clean_names = parse_annotator_names(",".join(annotator_names))
    clean_episodes = sorted({int(index) for index in episode_indices})
    if not clean_names:
        raise ValueError("至少需要一个标注员。")
    assignments = {name: [] for name in clean_names}
    for position, episode_index in enumerate(clean_episodes):
        annotator = clean_names[position % len(clean_names)]
        assignments[annotator].append(episode_index)
    return assignments


def normalize_assignments_payload(payload: Any) -> dict[str, list[int]]:
    """Normalize supported assignment JSON shapes."""
    if isinstance(payload, dict) and isinstance(payload.get("assignments"), dict):
        raw_assignments = payload["assignments"]
    else:
        raw_assignments = payload
    if not isinstance(raw_assignments, dict):
        raise ValueError("分配文件必须是对象，或包含 assignments 对象。")

    assignments: dict[str, list[int]] = {}
    for raw_name, raw_indices in raw_assignments.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if not isinstance(raw_indices, list):
            raise ValueError(f"标注员 {name} 的 episode 列表必须是数组。")
        episode_indices: list[int] = []
        seen: set[int] = set()
        for raw_index in raw_indices:
            try:
                episode_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"标注员 {name} 的 episode 编号无效: {raw_index}") from exc
            if episode_index < 0 or episode_index in seen:
                continue
            episode_indices.append(episode_index)
            seen.add(episode_index)
        assignments[name] = sorted(episode_indices)
    return assignments


def load_assignments(path: Path) -> dict[str, list[int]]:
    """Load assignments from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return normalize_assignments_payload(payload)


def save_assignments(path: Path, assignments: dict[str, list[int]]) -> None:
    """Write assignments in a human-editable JSON shape."""
    payload = {
        "version": 1,
        "assignments": {
            name: [int(index) for index in indices]
            for name, indices in sorted(assignments.items())
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def next_assigned_episode(
    assigned_episodes: list[int],
    *,
    current_episode: int,
    direction: int,
) -> int | None:
    """Return the nearest assigned episode in the requested direction."""
    episodes = sorted({int(index) for index in assigned_episodes})
    if not episodes:
        return None
    current_episode = int(current_episode)
    if direction >= 0:
        for episode_index in episodes:
            if episode_index > current_episode:
                return episode_index
        return episodes[-1]
    for episode_index in reversed(episodes):
        if episode_index < current_episode:
            return episode_index
    return episodes[0]
