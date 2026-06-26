"""Read LeRobot per-episode metadata from the dataset `meta/episodes/` parquet files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VIEW_KEYS = {
    "top": "videos/observation.images.top",
    "wrist": "videos/observation.images.right_wrist",
    "left_wrist": "videos/observation.images.left_wrist",
}


@dataclass(slots=True)
class EpisodeViewWindow:
    """One camera view's file + time range for a single episode."""

    file_index: int
    chunk_index: int
    from_seconds: float
    to_seconds: float


@dataclass(slots=True)
class EpisodeRange:
    """All information needed to load one episode across active views."""

    episode_index: int
    length_frames: int
    fps: float
    duration_seconds: float
    views: dict[str, EpisodeViewWindow] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeCatalog:
    """In-memory index of all episodes for a LeRobot dataset root."""

    dataset_root: Path
    fps: float
    total_episodes: int
    episodes: dict[int, EpisodeRange]

    def get(self, episode_index: int) -> EpisodeRange | None:
        """Return the episode metadata for a given index, or None."""
        return self.episodes.get(int(episode_index))


def load_episode_catalog(dataset_root: Path) -> EpisodeCatalog | None:
    """Read episode metadata from `meta/episodes/chunk-*/file-*.parquet`.

    Returns None when the dataset has no per-episode meta directory (older
    LeRobot layouts) or when pandas is unavailable.
    """
    resolved_root = dataset_root.expanduser().resolve()
    meta_dir = resolved_root / "meta" / "episodes"
    if not meta_dir.is_dir():
        return None

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    parquet_files = sorted(meta_dir.rglob("file-*.parquet"))
    if not parquet_files:
        return None

    fps = _read_fps(resolved_root)

    episodes: dict[int, EpisodeRange] = {}
    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            continue
        for _, row in df.iterrows():
            try:
                episode_index = int(row["episode_index"])
                length_frames = int(row["length"])
            except (KeyError, TypeError, ValueError):
                continue

            views: dict[str, EpisodeViewWindow] = {}
            for view_name, view_prefix in VIEW_KEYS.items():
                file_col = f"{view_prefix}/file_index"
                from_col = f"{view_prefix}/from_timestamp"
                to_col = f"{view_prefix}/to_timestamp"
                chunk_col = f"{view_prefix}/chunk_index"
                if file_col not in df.columns or from_col not in df.columns:
                    continue
                try:
                    views[view_name] = EpisodeViewWindow(
                        file_index=int(row[file_col]),
                        chunk_index=int(row[chunk_col]) if chunk_col in df.columns else 0,
                        from_seconds=float(row[from_col]),
                        to_seconds=float(row[to_col]),
                    )
                except (TypeError, ValueError):
                    continue

            duration_seconds = length_frames / fps if fps > 0 else 0.0
            episodes[episode_index] = EpisodeRange(
                episode_index=episode_index,
                length_frames=length_frames,
                fps=fps,
                duration_seconds=duration_seconds,
                views=views,
            )

    if not episodes:
        return None

    return EpisodeCatalog(
        dataset_root=resolved_root,
        fps=fps,
        total_episodes=len(episodes),
        episodes=episodes,
    )


def _read_fps(dataset_root: Path) -> float:
    """Read fps from `meta/info.json`, defaulting to 30.0."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    try:
        import json

        payload = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return float(payload.get("fps", 30.0))
    except Exception:
        pass
    return 30.0


def resolve_view_video_path(
    dataset_root: Path,
    view_name: str,
    chunk_index: int,
    file_index: int,
) -> Path | None:
    """Return the on-disk video file path for one view of one episode."""
    view_prefix = VIEW_KEYS.get(view_name)
    if view_prefix is None:
        return None
    # view_prefix begins with "videos/"
    relative = view_prefix.split("/", 1)[1]
    candidate = (
        dataset_root.expanduser().resolve()
        / "videos"
        / relative
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    if candidate.is_file():
        return candidate
    return None
