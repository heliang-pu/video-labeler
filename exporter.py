"""Export labeled frames as a LLaMA-Factory-compatible multimodal dataset."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import ProjectState


DEFAULT_VAL_SPLIT_RATIO = 0.1
DEFAULT_TEST_SPLIT_RATIO = 0.1
MIN_SAMPLES_FOR_SEPARATE_SPLITS = 5


class _SafeFormatDict(dict[str, object]):
    """Keep unknown placeholders intact instead of failing string formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(slots=True)
class ExportResult:
    """Metadata describing one finished dataset export."""

    output_dir: Path
    dataset_file: Path
    dataset_info_file: Path
    annotations_file: Path
    exported_samples: int
    split_files: dict[str, Path] = field(default_factory=dict)
    report_file: Path | None = None
    split_counts: dict[str, int] = field(default_factory=dict)
    training_config_file: Path | None = None
    training_readme_file: Path | None = None
    training_command: str | None = None
    training_config_error: str | None = None


@dataclass(slots=True)
class ExportSample:
    """Structured sample metadata used during export and split/report generation."""

    frame_index: int
    timestamp_seconds: float
    label: str
    image_relative_paths: list[str]
    record: dict[str, object]
    start_seconds: float | None = None
    end_seconds: float | None = None
    range_index: int | None = None


def sanitize_dataset_name(name: str) -> str:
    """Turn the UI dataset name into a stable LLaMA-Factory dataset key."""
    sanitized = re.sub(r"[^\w]+", "_", name.strip(), flags=re.UNICODE)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "video_labels"


def build_dataset_info(dataset_name: str, *, file_name: str = "dataset.json") -> dict[str, object]:
    """Return the dataset_info.json payload for LLaMA-Factory."""
    key = sanitize_dataset_name(dataset_name)
    return {
        key: {
            "file_name": file_name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "images": "images",
            },
        }
    }


def build_split_dataset_info(dataset_name: str, split_files: dict[str, str]) -> dict[str, object]:
    """Return dataset_info entries for the full dataset plus any exported splits."""
    dataset_info = build_dataset_info(dataset_name)
    dataset_key = sanitize_dataset_name(dataset_name)
    for split_name, split_file_name in split_files.items():
        dataset_info.update(
            build_dataset_info(
                f"{dataset_key}_{split_name}",
                file_name=split_file_name,
            )
        )
    return dataset_info


def build_dataset_record(
    project: ProjectState,
    *,
    frame_index: int,
    label: str,
    image_relative_paths: list[str],
    timestamp_seconds: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    range_index: int | None = None,
) -> dict[str, object]:
    """Build one single-record training sample."""
    video_name = project.video_path.stem if project.video_path is not None else "video"
    top_video_name = project.video_path.stem if project.video_path is not None else "top"
    wrist_video_name = (
        project.wrist_video_path.stem if project.wrist_video_path is not None else "wrist"
    )
    values = _SafeFormatDict(
        {
            "video_name": video_name,
            "top_video_name": top_video_name,
            "wrist_video_name": wrist_video_name,
            "frame_index": int(frame_index),
            "timestamp_seconds": float(timestamp_seconds),
            "start_seconds": float(timestamp_seconds if start_seconds is None else start_seconds),
            "end_seconds": float(timestamp_seconds if end_seconds is None else end_seconds),
            "range_index": -1 if range_index is None else int(range_index),
        }
    )
    input_text = project.input_template.format_map(values).strip()
    return {
        "instruction": project.instruction_template.strip(),
        "input": input_text,
        "output": label,
        "images": list(image_relative_paths),
    }


def _resolve_split_counts(total_samples: int) -> dict[str, int]:
    """Return train/val/test counts while keeping small datasets usable."""
    if total_samples <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if total_samples < MIN_SAMPLES_FOR_SEPARATE_SPLITS:
        return {"train": total_samples, "val": 0, "test": 0}

    val_count = max(1, int(round(total_samples * DEFAULT_VAL_SPLIT_RATIO)))
    test_count = max(1, int(round(total_samples * DEFAULT_TEST_SPLIT_RATIO)))
    max_eval_samples = max(0, total_samples - 1)

    while val_count + test_count > max_eval_samples:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    train_count = total_samples - val_count - test_count
    if train_count <= 0:
        return {"train": total_samples, "val": 0, "test": 0}
    return {"train": train_count, "val": val_count, "test": test_count}


def split_export_samples(samples: list[ExportSample]) -> dict[str, list[ExportSample]]:
    """Split sorted samples into temporal train/val/test segments."""
    counts = _resolve_split_counts(len(samples))
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_samples = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }
    return {split_name: split_items for split_name, split_items in split_samples.items() if split_items}


def _label_count_summary(labels: list[str]) -> dict[str, dict[str, float | int]]:
    """Build a count and ratio summary for one label list."""
    total_count = len(labels)
    counter = Counter(labels)
    summary: dict[str, dict[str, float | int]] = {}
    for label, count in sorted(counter.items()):
        ratio = 0.0 if total_count <= 0 else float(count) / float(total_count)
        summary[label] = {
            "count": int(count),
            "ratio": round(ratio, 6),
        }
    return summary


def _largest_unlabeled_gaps(
    labeled_frame_indices: list[int],
    *,
    frame_count: int,
    limit: int = 20,
) -> list[dict[str, int]]:
    """Return the largest unlabeled frame ranges in descending size order."""
    gaps: list[dict[str, int]] = []
    previous_index = -1
    for frame_index in labeled_frame_indices:
        gap_start = previous_index + 1
        gap_end = frame_index - 1
        if gap_end >= gap_start:
            gaps.append(
                {
                    "start_frame_index": gap_start,
                    "end_frame_index": gap_end,
                    "length": gap_end - gap_start + 1,
                }
            )
        previous_index = frame_index

    final_gap_start = previous_index + 1
    final_gap_end = frame_count - 1
    if final_gap_end >= final_gap_start:
        gaps.append(
            {
                "start_frame_index": final_gap_start,
                "end_frame_index": final_gap_end,
                "length": final_gap_end - final_gap_start + 1,
            }
        )

    gaps.sort(key=lambda item: item["length"], reverse=True)
    return gaps[:limit]


def build_export_report(
    project: ProjectState,
    *,
    frame_count: int,
    split_samples: dict[str, list[ExportSample]],
    all_samples: list[ExportSample],
) -> dict[str, object]:
    """Build a lightweight data-quality report for one export."""
    labeled_frame_indices = [sample.frame_index for sample in all_samples]
    label_counts = _label_count_summary([sample.label for sample in all_samples])

    split_summary: dict[str, object] = {}
    for split_name, samples in split_samples.items():
        split_summary[split_name] = {
            "sample_count": len(samples),
            "label_counts": _label_count_summary([sample.label for sample in samples]),
            "first_frame_index": samples[0].frame_index,
            "last_frame_index": samples[-1].frame_index,
        }

    average_gap = 0.0
    if len(labeled_frame_indices) >= 2:
        frame_gaps = [
            current_index - previous_index
            for previous_index, current_index in zip(
                labeled_frame_indices[:-1],
                labeled_frame_indices[1:],
            )
        ]
        average_gap = float(sum(frame_gaps)) / float(len(frame_gaps))

    coverage_ratio = 0.0
    if project.sublabels and frame_count > 0:
        if all(sample.start_seconds is not None and sample.end_seconds is not None for sample in all_samples):
            labeled_seconds = sum(
                max(0.0, float(sample.end_seconds) - float(sample.start_seconds))
                for sample in all_samples
            )
            fps = 0.0
            if labeled_frame_indices:
                last_frame_index = max(labeled_frame_indices)
                last_timestamp = max(sample.timestamp_seconds for sample in all_samples)
                if last_timestamp > 0:
                    fps = float(last_frame_index) / float(last_timestamp)
            total_seconds = 0.0 if fps <= 0 else float(frame_count) / fps
            if total_seconds > 0:
                coverage_ratio = labeled_seconds / total_seconds
    elif frame_count > 0:
        coverage_ratio = float(len(all_samples)) / float(frame_count)

    return {
        "video_path": None if project.video_path is None else str(project.video_path),
        "wrist_video_path": None
        if project.wrist_video_path is None
        else str(project.wrist_video_path),
        "left_wrist_video_path": None
        if project.left_wrist_video_path is None
        else str(project.left_wrist_video_path),
        "project_path": None if project.project_path is None else str(project.project_path),
        "dataset_name": sanitize_dataset_name(project.dataset_name),
        "label_group_name": project.label_group_name,
        "frame_count": int(frame_count),
        "labeled_sample_count": len(all_samples),
        "sublabel_count": len(project.sublabels),
        "coverage_ratio": round(coverage_ratio, 6),
        "frame_step": int(project.frame_step),
        "keyframe_threshold": float(project.keyframe_threshold),
        "average_labeled_frame_gap": round(average_gap, 3),
        "label_counts": label_counts,
        "split_summary": split_summary,
        "largest_unlabeled_gaps": _largest_unlabeled_gaps(
            labeled_frame_indices,
            frame_count=frame_count,
        ),
    }


def export_project_dataset(project: ProjectState, output_dir: Path) -> ExportResult:
    """Export labeled frames and dataset JSON files into one directory."""
    from .workspace_store import persist_project_workspace

    return persist_project_workspace(
        project,
        output_dir,
        require_annotations=True,
    )
