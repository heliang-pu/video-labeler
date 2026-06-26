"""Project persistence helpers for the video labeler."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import DEFAULT_LABEL_GROUP_NAME, DEFAULT_LABELS, ProjectState, TimeRangeLabel


PROJECT_VERSION = 5


def _relative_path(target: Path, base_dir: Path) -> str:
    """Encode a path relative to the target base directory."""
    return Path(os.path.relpath(target, base_dir)).as_posix()


def resolve_loaded_label_group_name(
    payload: dict[str, object],
    labels: list[str],
) -> str | None:
    """Resolve label-group metadata while keeping older project files compatible."""
    if "label_group_name" in payload:
        raw_group_name = payload.get("label_group_name")
        return None if raw_group_name is None else str(raw_group_name).strip() or None

    normalized_labels = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized_labels or normalized_labels == list(DEFAULT_LABELS):
        return DEFAULT_LABEL_GROUP_NAME
    return None


def save_project(project_file: Path, project: ProjectState) -> None:
    """Serialize the current project to a JSON file."""
    project_file = project_file.expanduser().resolve()
    project_file.parent.mkdir(parents=True, exist_ok=True)

    video_path = None
    if project.video_path is not None:
        video_path = _relative_path(project.video_path.expanduser().resolve(), project_file.parent)
    wrist_video_path = None
    if project.wrist_video_path is not None:
        wrist_video_path = _relative_path(
            project.wrist_video_path.expanduser().resolve(),
            project_file.parent,
        )
    left_wrist_video_path = None
    if project.left_wrist_video_path is not None:
        left_wrist_video_path = _relative_path(
            project.left_wrist_video_path.expanduser().resolve(),
            project_file.parent,
        )
    source_dataset_root = None
    if project.source_dataset_root is not None:
        source_dataset_root = _relative_path(
            project.source_dataset_root.expanduser().resolve(),
            project_file.parent,
        )

    payload = {
        "version": PROJECT_VERSION,
        "video_path": video_path,
        "wrist_video_path": wrist_video_path,
        "left_wrist_video_path": left_wrist_video_path,
        "video_segment_paths": [
            _relative_path(path.expanduser().resolve(), project_file.parent)
            for path in project.video_segment_paths
        ],
        "wrist_video_segment_paths": [
            _relative_path(path.expanduser().resolve(), project_file.parent)
            for path in project.wrist_video_segment_paths
        ],
        "left_wrist_video_segment_paths": [
            _relative_path(path.expanduser().resolve(), project_file.parent)
            for path in project.left_wrist_video_segment_paths
        ],
        "source_dataset_root": source_dataset_root,
        "labels": list(project.labels),
        "annotations": {str(index): label for index, label in sorted(project.annotations.items())},
        "sublabels": _serialize_sublabels(project.sublabels),
        "current_frame_index": int(project.current_frame_index),
        "frame_step": int(project.frame_step),
        "keyframe_threshold": float(project.keyframe_threshold),
        "label_group_name": project.label_group_name,
        "instruction_template": project.instruction_template,
        "input_template": project.input_template,
        "dataset_name": project.dataset_name,
    }
    _write_json_atomic(project_file, payload)


def load_project(project_file: Path) -> ProjectState:
    """Load a project JSON file and resolve any relative paths."""
    project_file = project_file.expanduser().resolve()
    payload = json.loads(project_file.read_text(encoding="utf-8"))

    raw_video_path = payload.get("video_path")
    video_path = None
    if raw_video_path:
        video_path = (project_file.parent / raw_video_path).resolve()
    raw_wrist_video_path = payload.get("wrist_video_path")
    wrist_video_path = None
    if raw_wrist_video_path:
        wrist_video_path = (project_file.parent / raw_wrist_video_path).resolve()
    raw_left_wrist_video_path = payload.get("left_wrist_video_path")
    left_wrist_video_path = None
    if raw_left_wrist_video_path:
        left_wrist_video_path = (project_file.parent / raw_left_wrist_video_path).resolve()
    raw_source_dataset_root = payload.get("source_dataset_root")
    source_dataset_root = None
    if raw_source_dataset_root:
        source_dataset_root = (project_file.parent / raw_source_dataset_root).resolve()

    project = ProjectState(
        video_path=video_path,
        wrist_video_path=wrist_video_path,
        left_wrist_video_path=left_wrist_video_path,
        video_segment_paths=_resolve_path_list(
            payload.get("video_segment_paths"),
            base_dir=project_file.parent,
        ),
        wrist_video_segment_paths=_resolve_path_list(
            payload.get("wrist_video_segment_paths"),
            base_dir=project_file.parent,
        ),
        left_wrist_video_segment_paths=_resolve_path_list(
            payload.get("left_wrist_video_segment_paths"),
            base_dir=project_file.parent,
        ),
        source_dataset_root=source_dataset_root,
        labels=list(payload.get("labels") or []),
        annotations={int(index): label for index, label in (payload.get("annotations") or {}).items()},
        sublabels=_load_sublabels(payload.get("sublabels")),
        current_frame_index=int(payload.get("current_frame_index", 0)),
        frame_step=max(1, int(payload.get("frame_step", 1))),
        keyframe_threshold=max(
            0.0,
            float(payload.get("keyframe_threshold", ProjectState().keyframe_threshold)),
        ),
        label_group_name=resolve_loaded_label_group_name(
            payload,
            list(payload.get("labels") or []),
        ),
        instruction_template=str(
            payload.get("instruction_template", ProjectState().instruction_template)
        ),
        input_template=str(payload.get("input_template", ProjectState().input_template)),
        dataset_name=str(payload.get("dataset_name", ProjectState().dataset_name)),
        project_path=project_file,
    )

    if not project.labels:
        project.labels = list(ProjectState().labels)

    return project


def _resolve_path_list(payload: object, *, base_dir: Path) -> list[Path]:
    """Resolve a serialized list of relative paths."""
    if not isinstance(payload, list):
        return []
    resolved_paths: list[Path] = []
    for raw_path in payload:
        if not raw_path:
            continue
        resolved_paths.append((base_dir / str(raw_path)).resolve())
    return resolved_paths


def _serialize_sublabels(sublabels: list[TimeRangeLabel]) -> list[dict[str, object]]:
    """Serialize time-range labels to JSON."""
    return [
        {
            "start_seconds": round(float(item.start_seconds), 6),
            "end_seconds": round(float(item.end_seconds), 6),
            "label": item.label,
        }
        for item in sublabels
    ]


def _load_sublabels(payload: object) -> list[TimeRangeLabel]:
    """Load saved time-range labels."""
    if not isinstance(payload, list):
        return []
    sublabels: list[TimeRangeLabel] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            sublabel = TimeRangeLabel(
                start_seconds=float(item.get("start_seconds", 0.0)),
                end_seconds=float(item.get("end_seconds", 0.0)),
                label=str(item.get("label", "")).strip(),
            ).normalized()
        except (TypeError, ValueError):
            continue
        if not sublabel.label or sublabel.end_seconds <= sublabel.start_seconds:
            continue
        sublabels.append(sublabel)
    return sorted(sublabels, key=lambda item: (item.start_seconds, item.end_seconds, item.label))


def _write_json_atomic(target_path: Path, payload: dict[str, object]) -> None:
    """Atomically replace one JSON file to reduce corruption during autosave."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_text = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
        text=True,
    )
    temp_path = Path(temp_path_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
