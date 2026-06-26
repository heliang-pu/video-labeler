"""Merge per-episode time-range labels into one dataset-level sidecar."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .episode_metadata import EpisodeCatalog
from .models import TimeRangeLabel
from .workspace_store import SUBLABELS_FILE_NAME, _load_sublabels, _write_json_atomic


@dataclass(slots=True)
class EpisodeMergeResult:
    """Summary of one multi-episode sublabel merge."""

    output_file: Path
    merged_episode_count: int
    missing_episode_indices: list[int]
    total_sublabel_count: int
    label_counts: dict[str, int]


def merge_episode_sublabels(
    dataset_root: Path,
    catalog: EpisodeCatalog,
) -> EpisodeMergeResult:
    """Merge `sublabels_episode_*.json` into dataset-root `sublabels.json`."""
    dataset_root = dataset_root.expanduser().resolve()
    merged_ranges: list[dict[str, object]] = []
    label_counter: Counter[str] = Counter()
    merged_episode_indices: set[int] = set()
    missing_episode_indices: list[int] = []

    for episode_index in sorted(catalog.episodes):
        episode = catalog.episodes[episode_index]
        sidecar_path = dataset_root / f"sublabels_episode_{episode_index:04d}.json"
        if not sidecar_path.exists():
            missing_episode_indices.append(episode_index)
            continue
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sublabels = _load_sublabels(payload)
        if not sublabels:
            missing_episode_indices.append(episode_index)
            continue

        episode_offset_seconds = _episode_global_offset_seconds(catalog, episode_index)
        for sublabel in sublabels:
            bounded = _bound_sublabel_to_episode(sublabel, episode.duration_seconds)
            if bounded is None:
                continue
            start_seconds = episode_offset_seconds + bounded.start_seconds
            end_seconds = episode_offset_seconds + bounded.end_seconds
            merged_ranges.append(
                {
                    "episode_index": int(episode_index),
                    "start_seconds": round(start_seconds, 6),
                    "end_seconds": round(end_seconds, 6),
                    "label": bounded.label,
                }
            )
            label_counter[bounded.label] += 1
        merged_episode_indices.add(episode_index)

    merged_ranges.sort(
        key=lambda item: (
            float(item["start_seconds"]),
            float(item["end_seconds"]),
            str(item["label"]),
        )
    )
    output_file = dataset_root / SUBLABELS_FILE_NAME
    payload = {
        "version": 1,
        "source_dataset_root": str(dataset_root),
        "dataset_name": dataset_root.name,
        "fps": float(catalog.fps),
        "episode_count": int(catalog.total_episodes),
        "merged_episode_count": len(merged_episode_indices),
        "missing_episode_indices": missing_episode_indices,
        "label_counts": dict(sorted(label_counter.items())),
        "sublabels": merged_ranges,
    }
    _write_json_atomic(output_file, payload)

    return EpisodeMergeResult(
        output_file=output_file,
        merged_episode_count=len(merged_episode_indices),
        missing_episode_indices=missing_episode_indices,
        total_sublabel_count=len(merged_ranges),
        label_counts=dict(sorted(label_counter.items())),
    )


def _episode_global_offset_seconds(catalog: EpisodeCatalog, episode_index: int) -> float:
    """Return offset in the concatenated logical dataset timeline."""
    offset_frames = 0
    for index in sorted(catalog.episodes):
        if index >= episode_index:
            break
        offset_frames += int(catalog.episodes[index].length_frames)
    if catalog.fps <= 0:
        return 0.0
    return float(offset_frames) / float(catalog.fps)


def _bound_sublabel_to_episode(
    sublabel: TimeRangeLabel,
    duration_seconds: float,
) -> TimeRangeLabel | None:
    """Clamp a sublabel to its episode bounds."""
    normalized = sublabel.normalized()
    start_seconds = min(max(0.0, normalized.start_seconds), duration_seconds)
    end_seconds = min(max(0.0, normalized.end_seconds), duration_seconds)
    if end_seconds <= start_seconds or not normalized.label:
        return None
    return TimeRangeLabel(start_seconds=start_seconds, end_seconds=end_seconds, label=normalized.label)
