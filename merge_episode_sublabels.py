"""CLI for merging per-episode sublabel files into one dataset sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from .collaboration_merge import merge_episode_sublabels
from .episode_metadata import load_episode_catalog


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge sublabels_episode_XXXX.json files under one LeRobot dataset root "
            "into sublabels.json."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="LeRobot dataset root, for example /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376",
    )
    return parser


def main() -> None:
    """Merge per-episode sublabels and print a compact summary."""
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    catalog = load_episode_catalog(dataset_root)
    if catalog is None:
        raise SystemExit(f"未找到 episode 元数据，无法合并: {dataset_root}")

    result = merge_episode_sublabels(dataset_root, catalog)
    label_text = ", ".join(
        f"{label}:{count}" for label, count in result.label_counts.items()
    ) or "无"
    missing_preview = ", ".join(str(index) for index in result.missing_episode_indices[:30])
    if len(result.missing_episode_indices) > 30:
        missing_preview += ", ..."
    missing_text = missing_preview or "无"

    print(f"输出: {result.output_file}")
    print(f"已合并 episode: {result.merged_episode_count}/{catalog.total_episodes}")
    print(f"区间标签数: {result.total_sublabel_count}")
    print(f"标签统计: {label_text}")
    print(f"未标/空标 episode: {missing_text}")


if __name__ == "__main__":
    main()
