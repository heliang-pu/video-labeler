"""Durable dataset-folder persistence for the Gradio video labeler."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    cv2 = None

from .exporter import (
    ExportResult,
    ExportSample,
    build_dataset_record,
    build_export_report,
    build_split_dataset_info,
    split_export_samples,
)
from .models import ProjectState
from .models import TimeRangeLabel
from .storage import resolve_loaded_label_group_name
from .time_ranges import frame_indices_for_seconds_range
from .training_config import generate_training_files
from .video import VideoDependencyError, VideoFrameReader


WORKSPACE_META_VERSION = 3
WORKSPACE_META_FILE_NAME = "workspace_meta.json"
DATASET_FILE_NAME = "dataset.json"
DATASET_INFO_FILE_NAME = "dataset_info.json"
ANNOTATIONS_FILE_NAME = "annotations.json"
SUBLABELS_FILE_NAME = "sublabels.json"
EXPORT_REPORT_FILE_NAME = "export_report.json"
IMAGES_DIR_NAME = "images"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(slots=True)
class WorkspaceLoadResult:
    """One dataset-folder restore result."""

    project: ProjectState
    dataset_dir: Path
    restored_from_workspace: bool
    last_labeled_frame_index: int | None


def workspace_has_state(dataset_dir: Path) -> bool:
    """Return whether the target directory already looks like a saved workspace."""
    resolved_dir = dataset_dir.expanduser().resolve()
    return any(
        (resolved_dir / file_name).exists()
        for file_name in (
            WORKSPACE_META_FILE_NAME,
            ANNOTATIONS_FILE_NAME,
            SUBLABELS_FILE_NAME,
        )
    )


def load_project_from_dataset_dir(
    dataset_dir: Path,
    *,
    video_path: Path | None = None,
    wrist_video_path: Path | None = None,
    left_wrist_video_path: Path | None = None,
    video_segment_paths: list[Path] | None = None,
    wrist_video_segment_paths: list[Path] | None = None,
    left_wrist_video_segment_paths: list[Path] | None = None,
    source_dataset_root: Path | None = None,
) -> WorkspaceLoadResult:
    """Restore project state from a dataset workspace directory."""
    dataset_dir = dataset_dir.expanduser().resolve()
    meta_path = dataset_dir / WORKSPACE_META_FILE_NAME
    annotations_path = dataset_dir / ANNOTATIONS_FILE_NAME
    sublabels_path = dataset_dir / SUBLABELS_FILE_NAME

    meta_payload: dict[str, Any] = {}
    if meta_path.exists():
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw_meta, dict):
            meta_payload = raw_meta

    annotations_payload: Any = []
    if annotations_path.exists():
        annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    sublabels_payload: Any = {}
    if sublabels_path.exists():
        sublabels_payload = json.loads(sublabels_path.read_text(encoding="utf-8"))

    annotations = _load_annotations(annotations_payload)
    sublabels = _load_sublabels(sublabels_payload)
    labels = _load_labels(meta_payload, annotations_payload, annotations, sublabels)
    label_group_name = resolve_loaded_label_group_name(meta_payload, labels)

    stored_video_path = None
    raw_video_path = meta_payload.get("video_path")
    if raw_video_path:
        stored_video_path = (dataset_dir / str(raw_video_path)).expanduser().resolve()
    stored_wrist_video_path = None
    raw_wrist_video_path = meta_payload.get("wrist_video_path")
    if raw_wrist_video_path:
        stored_wrist_video_path = (dataset_dir / str(raw_wrist_video_path)).expanduser().resolve()
    stored_left_wrist_video_path = None
    raw_left_wrist_video_path = meta_payload.get("left_wrist_video_path")
    if raw_left_wrist_video_path:
        stored_left_wrist_video_path = (
            dataset_dir / str(raw_left_wrist_video_path)
        ).expanduser().resolve()
    stored_source_dataset_root = None
    raw_source_dataset_root = meta_payload.get("source_dataset_root")
    if raw_source_dataset_root:
        stored_source_dataset_root = (
            dataset_dir / str(raw_source_dataset_root)
        ).expanduser().resolve()

    resolved_video_path = stored_video_path
    resolved_wrist_video_path = stored_wrist_video_path
    resolved_left_wrist_video_path = stored_left_wrist_video_path
    if video_path is not None:
        resolved_video_path = video_path.expanduser().resolve()
        if stored_video_path is not None and stored_video_path != resolved_video_path:
            raise ValueError(
                "当前数据集目录绑定的视频与输入视频不一致，"
                f" 工作区视频: {stored_video_path}，输入视频: {resolved_video_path}"
            )
    if wrist_video_path is not None:
        resolved_wrist_video_path = wrist_video_path.expanduser().resolve()
        if (
            stored_wrist_video_path is not None
            and stored_wrist_video_path != resolved_wrist_video_path
        ):
            raise ValueError(
                "当前数据集目录绑定的 wrist 视频与输入视频不一致，"
                f" 工作区 wrist: {stored_wrist_video_path}，输入 wrist: {resolved_wrist_video_path}"
            )
    if left_wrist_video_path is not None:
        resolved_left_wrist_video_path = left_wrist_video_path.expanduser().resolve()
        if (
            stored_left_wrist_video_path is not None
            and stored_left_wrist_video_path != resolved_left_wrist_video_path
        ):
            raise ValueError(
                "当前数据集目录绑定的 left_wrist 视频与输入视频不一致，"
                f" 工作区 left_wrist: {stored_left_wrist_video_path}，"
                f"输入 left_wrist: {resolved_left_wrist_video_path}"
            )
    if stored_wrist_video_path is not None and wrist_video_path is None and video_path is not None:
        raise ValueError(
            "当前数据集目录绑定了 wrist 视频，请同时提供 wrist 视频路径后再恢复工作区。"
        )
    if (
        "wrist_video_path" in meta_payload
        and stored_wrist_video_path is None
        and wrist_video_path is not None
    ):
        raise ValueError(
            "当前数据集目录是单视角工作区，输入却包含了额外的 wrist 视频。"
        )
    if stored_left_wrist_video_path is not None and left_wrist_video_path is None and video_path is not None:
        raise ValueError(
            "当前数据集目录绑定了 left_wrist 视频，请同时提供 left_wrist 视频路径后再恢复工作区。"
        )
    if (
        "left_wrist_video_path" in meta_payload
        and stored_left_wrist_video_path is None
        and left_wrist_video_path is not None
    ):
        raise ValueError(
            "当前数据集目录没有绑定 left_wrist 视频，输入却包含了额外的 left_wrist 视频。"
        )

    resolved_video_segments = _resolve_segments_from_inputs(
        video_segment_paths,
        stored_payload=meta_payload.get("video_segment_paths"),
        base_dir=dataset_dir,
        fallback=resolved_video_path,
    )
    resolved_wrist_segments = _resolve_segments_from_inputs(
        wrist_video_segment_paths,
        stored_payload=meta_payload.get("wrist_video_segment_paths"),
        base_dir=dataset_dir,
        fallback=resolved_wrist_video_path,
    )
    resolved_left_wrist_segments = _resolve_segments_from_inputs(
        left_wrist_video_segment_paths,
        stored_payload=meta_payload.get("left_wrist_video_segment_paths"),
        base_dir=dataset_dir,
        fallback=resolved_left_wrist_video_path,
    )
    resolved_source_dataset_root = (
        source_dataset_root.expanduser().resolve()
        if source_dataset_root is not None
        else stored_source_dataset_root
    )

    dataset_name = str(meta_payload.get("dataset_name") or dataset_dir.name or ProjectState().dataset_name)
    frame_step = max(1, int(meta_payload.get("frame_step", ProjectState().frame_step)))
    keyframe_threshold = max(
        0.0,
        float(meta_payload.get("keyframe_threshold", ProjectState().keyframe_threshold)),
    )
    current_frame_index = _resolve_current_frame_index(meta_payload, annotations)

    raw_episode_index = meta_payload.get("episode_index")
    stored_episode_index: int | None = None
    if raw_episode_index is not None:
        try:
            stored_episode_index = int(raw_episode_index)
        except (TypeError, ValueError):
            stored_episode_index = None

    project = ProjectState(
        video_path=resolved_video_path,
        wrist_video_path=resolved_wrist_video_path,
        left_wrist_video_path=resolved_left_wrist_video_path,
        video_segment_paths=resolved_video_segments,
        wrist_video_segment_paths=resolved_wrist_segments,
        left_wrist_video_segment_paths=resolved_left_wrist_segments,
        source_dataset_root=resolved_source_dataset_root,
        labels=labels,
        annotations=annotations,
        sublabels=sublabels,
        current_frame_index=current_frame_index,
        frame_step=frame_step,
        keyframe_threshold=keyframe_threshold,
        label_group_name=label_group_name,
        instruction_template=str(
            meta_payload.get("instruction_template", ProjectState().instruction_template)
        ),
        input_template=str(meta_payload.get("input_template", ProjectState().input_template)),
        dataset_name=dataset_name,
        episode_index=stored_episode_index,
    )
    return WorkspaceLoadResult(
        project=project,
        dataset_dir=dataset_dir,
        restored_from_workspace=bool(meta_payload or annotations or sublabels),
        last_labeled_frame_index=max(annotations) if annotations else None,
    )


def persist_project_workspace(
    project: ProjectState,
    output_dir: Path,
    *,
    reader: VideoFrameReader | None = None,
    require_annotations: bool = False,
    progress_callback=None,
) -> ExportResult:
    """Write the current project into one LLaMA-Factory dataset workspace directory."""
    if project.video_path is None:
        raise ValueError("Please load a video before saving the dataset workspace.")
    if require_annotations and not project.annotations and not project.sublabels:
        raise ValueError("No labels found. Add at least one time range before exporting.")
    if cv2 is None:
        raise VideoDependencyError(
            "OpenCV is not installed. Install dependencies from requirements.txt first."
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / IMAGES_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = output_dir / DATASET_FILE_NAME
    dataset_info_file = output_dir / DATASET_INFO_FILE_NAME
    annotations_file = output_dir / ANNOTATIONS_FILE_NAME
    sublabels_file = output_dir / SUBLABELS_FILE_NAME
    report_file = output_dir / EXPORT_REPORT_FILE_NAME
    workspace_meta_file = output_dir / WORKSPACE_META_FILE_NAME
    source_sublabels_file = _source_sublabels_path(project)
    training_config_file = None
    training_readme_file = None
    training_command = None
    training_config_error = None

    active_reader, should_close_reader = _ensure_reader(
        reader,
        project.video_path,
        project.wrist_video_path,
        project.left_wrist_video_path,
        video_segment_paths=project.video_segment_paths,
        wrist_video_segment_paths=project.wrist_video_segment_paths,
        left_wrist_video_segment_paths=project.left_wrist_video_segment_paths,
    )
    try:
        if progress_callback:
            progress_callback(0, 100, "验证标签...")
        _validate_annotation_bounds(project, frame_count=active_reader.frame_count)
        _validate_sublabel_bounds(project, reader=active_reader)

        if progress_callback:
            progress_callback(5, 100, "提取帧图像...")
        samples, annotations_summary = _build_samples(
            project,
            output_dir=output_dir,
            images_dir=images_dir,
            reader=active_reader,
            progress_callback=progress_callback,
        )
        if progress_callback:
            progress_callback(85, 100, "生成数据集文件...")
        split_samples = split_export_samples(samples)
        split_counts = {
            split_name: len(split_items)
            for split_name, split_items in split_samples.items()
        }

        dataset_records = [sample.record for sample in samples]
        _write_json_atomic(dataset_file, dataset_records)

        split_files: dict[str, Path] = {}
        split_file_names: dict[str, str] = {}
        for split_name, split_items in split_samples.items():
            split_path = output_dir / f"{split_name}.json"
            _write_json_atomic(split_path, [sample.record for sample in split_items])
            split_files[split_name] = split_path
            split_file_names[split_name] = split_path.name

        if progress_callback:
            progress_callback(90, 100, "清理旧文件...")
        _delete_unused_split_files(output_dir, split_files)
        _delete_stale_images(images_dir, samples)

        if progress_callback:
            progress_callback(95, 100, "写入元数据...")
        dataset_info_payload = build_split_dataset_info(project.dataset_name, split_file_names)
        _write_json_atomic(
            dataset_info_file,
            dataset_info_payload,
        )
        _write_json_atomic(
            annotations_file,
            annotations_summary,
        )
        sublabels_payload = _build_sublabels_payload(project, active_reader)
        _write_json_atomic(sublabels_file, sublabels_payload)
        if source_sublabels_file is not None:
            _write_json_atomic(source_sublabels_file, sublabels_payload)
        _write_json_atomic(
            workspace_meta_file,
            _build_workspace_meta(project, output_dir=output_dir),
        )
        _write_json_atomic(
            report_file,
            build_export_report(
                project,
                frame_count=active_reader.frame_count,
                split_samples=split_samples,
                all_samples=samples,
            ),
        )
        try:
            training_result = generate_training_files(
                output_dir=output_dir,
                dataset_info=dataset_info_payload,
            )
        except Exception as exc:
            training_config_error = str(exc)
        else:
            training_config_file = training_result.config_file
            training_readme_file = training_result.readme_file
            training_command = training_result.command
    finally:
        if should_close_reader:
            active_reader.close()

    return ExportResult(
        output_dir=output_dir,
        dataset_file=dataset_file,
        dataset_info_file=dataset_info_file,
        annotations_file=annotations_file,
        split_files=split_files,
        report_file=report_file,
        split_counts=split_counts,
        exported_samples=len(samples),
        training_config_file=training_config_file,
        training_readme_file=training_readme_file,
        training_command=training_command,
        training_config_error=training_config_error,
    )


def _ensure_reader(
    reader: VideoFrameReader | None,
    video_path: Path,
    wrist_video_path: Path | None,
    left_wrist_video_path: Path | None = None,
    *,
    video_segment_paths: list[Path] | None = None,
    wrist_video_segment_paths: list[Path] | None = None,
    left_wrist_video_segment_paths: list[Path] | None = None,
) -> tuple[VideoFrameReader, bool]:
    """Return a reader opened on the target video and whether it should be closed later."""
    resolved_video_path = video_path.expanduser().resolve()
    resolved_wrist_video_path = None
    if wrist_video_path is not None:
        resolved_wrist_video_path = wrist_video_path.expanduser().resolve()
    resolved_left_wrist_video_path = None
    if left_wrist_video_path is not None:
        resolved_left_wrist_video_path = left_wrist_video_path.expanduser().resolve()
    if (
        reader is not None
        and reader.matches_paths(
            resolved_video_path,
            resolved_wrist_video_path,
            resolved_left_wrist_video_path,
            video_segment_paths=video_segment_paths,
            wrist_video_segment_paths=wrist_video_segment_paths,
            left_wrist_video_segment_paths=left_wrist_video_segment_paths,
        )
    ):
        return reader, False

    temp_reader = VideoFrameReader()
    temp_reader.open(
        resolved_video_path,
        resolved_wrist_video_path,
        resolved_left_wrist_video_path,
        video_segment_paths=video_segment_paths,
        wrist_video_segment_paths=wrist_video_segment_paths,
        left_wrist_video_segment_paths=left_wrist_video_segment_paths,
    )
    return temp_reader, True


def _build_samples(
    project: ProjectState,
    *,
    output_dir: Path,
    images_dir: Path,
    reader: VideoFrameReader,
    progress_callback=None,
) -> tuple[list[ExportSample], list[dict[str, object]]]:
    """Build dataset records and sidecar annotation summaries."""
    samples: list[ExportSample] = []
    annotations_summary: list[dict[str, object]] = []
    if project.sublabels:
        return _build_sublabel_samples(
            project,
            output_dir=output_dir,
            images_dir=images_dir,
            reader=reader,
            progress_callback=progress_callback,
        )

    total_annotations = len(project.annotations)
    for idx, (frame_index, label) in enumerate(sorted(project.annotations.items())):
        if progress_callback and total_annotations > 0:
            progress_pct = int(5 + (idx / total_annotations) * 80)
            progress_callback(progress_pct, 100, f"处理帧 {idx + 1}/{total_annotations}...")
        image_relative_paths: list[str] = []
        view_images: dict[str, str] = {}
        frames_by_view: dict[str, Any] | None = None
        for view_name, view_path in project.active_video_items():
            view_images_dir = images_dir / view_name
            view_images_dir.mkdir(parents=True, exist_ok=True)
            image_name = f"{view_path.stem}_{frame_index:06d}.jpg"
            image_path = view_images_dir / image_name
            if not image_path.exists():
                if frames_by_view is None:
                    frames_by_view = reader.read_frames_bgr(frame_index)
                frame_bgr = frames_by_view[view_name]
                if not cv2.imwrite(str(image_path), frame_bgr):
                    raise RuntimeError(f"Failed to write extracted frame: {image_path}")

            image_relative_path = Path(os.path.relpath(image_path, output_dir)).as_posix()
            image_relative_paths.append(image_relative_path)
            view_images[view_name] = image_relative_path

        timestamp_seconds = reader.timestamp_seconds(frame_index)
        record = build_dataset_record(
            project,
            frame_index=frame_index,
            label=label,
            image_relative_paths=image_relative_paths,
            timestamp_seconds=timestamp_seconds,
        )
        samples.append(
            ExportSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                label=label,
                image_relative_paths=image_relative_paths,
                record=record,
            )
        )
        annotations_summary.append(
            {
                "frame_index": int(frame_index),
                "timestamp_seconds": round(timestamp_seconds, 6),
                "label": label,
                "images": image_relative_paths,
                "view_images": view_images,
            }
        )
    return samples, annotations_summary


def _build_sublabel_samples(
    project: ProjectState,
    *,
    output_dir: Path,
    images_dir: Path,
    reader: VideoFrameReader,
    progress_callback=None,
) -> tuple[list[ExportSample], list[dict[str, object]]]:
    """Build VLM training samples for every frame covered by each time-range label."""
    samples: list[ExportSample] = []
    annotations_summary: list[dict[str, object]] = []
    active_video_items = project.active_video_items()
    total_sublabels = len(project.sublabels)

    range_frame_indices = [
        frame_indices_for_seconds_range(
            sublabel.start_seconds,
            sublabel.end_seconds,
            fps=reader.fps,
            frame_count=reader.frame_count,
        )
        for sublabel in project.sublabels
    ]
    total_samples = sum(len(frame_indices) for frame_indices in range_frame_indices)

    sample_count = 0
    for range_index, sublabel in enumerate(project.sublabels):
        for sample_index, frame_index in enumerate(range_frame_indices[range_index]):
            if progress_callback and total_samples > 0:
                progress_pct = int(5 + (sample_count / total_samples) * 80)
                progress_callback(
                    progress_pct,
                    100,
                    f"处理时间段 {range_index + 1}/{total_sublabels}，"
                    f"帧 {sample_count + 1}/{total_samples}...",
                )
            sample_count += 1

            image_relative_paths: list[str] = []
            view_images: dict[str, str] = {}
            frames_by_view: dict[str, Any] | None = None
            segment_index, local_frame_index = reader.segment_for_frame(frame_index)

            for view_name, view_path in active_video_items:
                view_images_dir = images_dir / view_name
                view_images_dir.mkdir(parents=True, exist_ok=True)
                image_name = (
                    f"range_{range_index:04d}_sample_{sample_index:03d}_{view_path.stem}"
                    f"_seg{segment_index:03d}_frame{local_frame_index:06d}.jpg"
                )
                image_path = view_images_dir / image_name
                if not image_path.exists():
                    if frames_by_view is None:
                        frames_by_view = reader.read_frames_bgr(frame_index)
                    frame_bgr = frames_by_view[view_name]
                    if not cv2.imwrite(str(image_path), frame_bgr):
                        raise RuntimeError(f"Failed to write extracted frame: {image_path}")

                image_relative_path = Path(os.path.relpath(image_path, output_dir)).as_posix()
                image_relative_paths.append(image_relative_path)
                view_images[view_name] = image_relative_path

            timestamp_seconds = reader.timestamp_seconds(frame_index)
            record = build_dataset_record(
                project,
                frame_index=frame_index,
                label=sublabel.label,
                image_relative_paths=image_relative_paths,
                timestamp_seconds=timestamp_seconds,
                start_seconds=sublabel.start_seconds,
                end_seconds=sublabel.end_seconds,
                range_index=range_index,
            )
            samples.append(
                ExportSample(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    label=sublabel.label,
                    image_relative_paths=image_relative_paths,
                    record=record,
                    start_seconds=sublabel.start_seconds,
                    end_seconds=sublabel.end_seconds,
                    range_index=range_index,
                )
            )
            annotations_summary.append(
                {
                    "range_index": int(range_index),
                    "sample_index": int(sample_index),
                    "start_seconds": round(sublabel.start_seconds, 6),
                    "end_seconds": round(sublabel.end_seconds, 6),
                    "timestamp_seconds": round(timestamp_seconds, 6),
                    "frame_index": int(frame_index),
                    "segment_index": int(segment_index),
                    "segment_frame_index": int(local_frame_index),
                    "label": sublabel.label,
                    "images": image_relative_paths,
                    "view_images": view_images,
                }
            )

    return samples, annotations_summary


def _validate_annotation_bounds(project: ProjectState, *, frame_count: int) -> None:
    """Fail fast when saved annotations point outside the loaded video."""
    invalid_indices = [
        frame_index
        for frame_index in sorted(project.annotations)
        if frame_index < 0 or frame_index >= frame_count
    ]
    if not invalid_indices:
        return

    preview = ", ".join(str(frame_index) for frame_index in invalid_indices[:10])
    if len(invalid_indices) > 10:
        preview += ", ..."
    raise ValueError(
        "存在超出当前视频范围的标注帧，无法保存数据集。"
        f" 越界帧: {preview}，视频总帧数: {frame_count}"
    )


def _validate_sublabel_bounds(project: ProjectState, *, reader: VideoFrameReader) -> None:
    """Fail fast when saved sublabels point outside the logical video timeline."""
    if not project.sublabels:
        return
    if reader.frame_count <= 0 or reader.fps <= 0:
        raise ValueError("无法获取视频总时长，不能保存秒级区间标签。")
    total_seconds = reader.frame_count / reader.fps
    invalid_items = [
        index
        for index, item in enumerate(project.sublabels)
        if item.start_seconds < 0
        or item.end_seconds <= item.start_seconds
        or item.end_seconds > total_seconds + 1e-6
        or not item.label.strip()
    ]
    if not invalid_items:
        return
    preview = ", ".join(str(index) for index in invalid_items[:10])
    if len(invalid_items) > 10:
        preview += ", ..."
    raise ValueError(
        "存在超出当前视频范围或无效的区间标签，无法保存数据集。"
        f" 区间序号: {preview}，视频总时长: {total_seconds:.3f}s"
    )


def _seconds_to_frame_index(seconds: float, *, reader: VideoFrameReader) -> int:
    """Convert seconds on the logical timeline to a bounded frame index."""
    if reader.frame_count <= 0:
        return 0
    if reader.fps <= 0:
        return 0
    raw_index = int(round(max(0.0, float(seconds)) * reader.fps))
    return max(0, min(raw_index, reader.frame_count - 1))


def _build_workspace_meta(project: ProjectState, *, output_dir: Path) -> dict[str, object]:
    """Serialize non-training project state used for resume and editing."""
    video_path = None
    if project.video_path is not None:
        video_path = Path(os.path.relpath(project.video_path.expanduser().resolve(), output_dir)).as_posix()
    wrist_video_path = None
    if project.wrist_video_path is not None:
        wrist_video_path = Path(
            os.path.relpath(project.wrist_video_path.expanduser().resolve(), output_dir)
        ).as_posix()
    left_wrist_video_path = None
    if project.left_wrist_video_path is not None:
        left_wrist_video_path = Path(
            os.path.relpath(project.left_wrist_video_path.expanduser().resolve(), output_dir)
        ).as_posix()
    source_dataset_root = None
    if project.source_dataset_root is not None:
        source_dataset_root = Path(
            os.path.relpath(project.source_dataset_root.expanduser().resolve(), output_dir)
        ).as_posix()

    labeled_frame_indices = sorted(project.annotations)
    return {
        "version": WORKSPACE_META_VERSION,
        "video_path": video_path,
        "wrist_video_path": wrist_video_path,
        "left_wrist_video_path": left_wrist_video_path,
        "video_segment_paths": [
            Path(os.path.relpath(path.expanduser().resolve(), output_dir)).as_posix()
            for path in project.video_segment_paths
        ],
        "wrist_video_segment_paths": [
            Path(os.path.relpath(path.expanduser().resolve(), output_dir)).as_posix()
            for path in project.wrist_video_segment_paths
        ],
        "left_wrist_video_segment_paths": [
            Path(os.path.relpath(path.expanduser().resolve(), output_dir)).as_posix()
            for path in project.left_wrist_video_segment_paths
        ],
        "source_dataset_root": source_dataset_root,
        "dataset_name": project.dataset_name,
        "labels": list(project.labels),
        "annotations_count": len(project.annotations),
        "sublabels_count": len(project.sublabels),
        "current_frame_index": int(project.current_frame_index),
        "last_labeled_frame_index": labeled_frame_indices[-1] if labeled_frame_indices else None,
        "frame_step": int(project.frame_step),
        "keyframe_threshold": float(project.keyframe_threshold),
        "label_group_name": project.label_group_name,
        "instruction_template": project.instruction_template,
        "input_template": project.input_template,
        "episode_index": project.episode_index,
    }


def _delete_unused_split_files(output_dir: Path, split_files: dict[str, Path]) -> None:
    """Delete train/val/test JSON files that are no longer generated."""
    active_file_names = {path.name for path in split_files.values()}
    for split_name in SPLIT_NAMES:
        split_path = output_dir / f"{split_name}.json"
        if split_path.exists() and split_path.name not in active_file_names:
            split_path.unlink()


def _delete_stale_images(images_dir: Path, samples: list[ExportSample]) -> None:
    """Delete extracted frame images that no longer correspond to labeled frames."""
    expected_relative_paths = {
        image_relative_path
        for sample in samples
        for image_relative_path in sample.image_relative_paths
    }
    for image_path in sorted(
        images_dir.rglob("*"),
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    ):
        if image_path.is_file():
            image_relative_path = Path(os.path.relpath(image_path, images_dir.parent)).as_posix()
            if image_relative_path not in expected_relative_paths:
                image_path.unlink()
            continue
        if image_path.is_dir() and not any(image_path.iterdir()):
            image_path.rmdir()


def _load_annotations(payload: Any) -> dict[int, str]:
    """Load saved frame annotations from either dict or list format."""
    annotations: dict[int, str] = {}
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            items.append((item.get("frame_index"), item.get("label")))
    else:
        items = []

    for raw_index, raw_label in items:
        try:
            frame_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        label = str(raw_label).strip()
        if not label:
            continue
        annotations[frame_index] = label
    return dict(sorted(annotations.items()))


def _load_sublabels(payload: Any) -> list[TimeRangeLabel]:
    """Load saved time-range labels from the dataset-root sidecar JSON."""
    if isinstance(payload, dict):
        raw_sublabels = payload.get("sublabels")
    else:
        raw_sublabels = payload
    if not isinstance(raw_sublabels, list):
        raw_sublabels = payload.get("time_ranges") if isinstance(payload, dict) else []
    if not isinstance(raw_sublabels, list):
        return []

    sublabels: list[TimeRangeLabel] = []
    for item in raw_sublabels:
        if not isinstance(item, dict):
            continue
        try:
            start_value = item.get("start_seconds", item.get("start_sec", 0.0))
            end_value = item.get("end_seconds", item.get("end_sec", 0.0))
            sublabel = TimeRangeLabel(
                start_seconds=float(start_value),
                end_seconds=float(end_value),
                label=str(item.get("label", "")).strip(),
            ).normalized()
        except (TypeError, ValueError):
            continue
        if not sublabel.label or sublabel.end_seconds <= sublabel.start_seconds:
            continue
        sublabels.append(sublabel)
    return sorted(sublabels, key=lambda item: (item.start_seconds, item.end_seconds, item.label))


def _build_sublabels_payload(project: ProjectState, reader: VideoFrameReader) -> dict[str, object]:
    """Build the resumable sidecar JSON stored beside the source dataset."""
    total_seconds = reader.frame_count / reader.fps if reader.frame_count > 0 and reader.fps > 0 else 0.0
    segment_payload = reader.segment_metadata()

    return {
        "version": 1,
        "source_dataset_root": None
        if project.source_dataset_root is None
        else str(project.source_dataset_root),
        "dataset_name": project.dataset_name,
        "labels": list(project.labels),
        "views": list(reader.active_views),
        "fps": float(reader.fps),
        "frame_count": int(reader.frame_count),
        "total_seconds": round(total_seconds, 6),
        "segments": segment_payload,
        "video_paths": {
            view_name: [str(path) for path in paths]
            for view_name, paths in reader.active_video_segments().items()
        },
        "current_frame_index": int(project.current_frame_index),
        "sublabels": [
            {
                "start_seconds": round(item.start_seconds, 6),
                "end_seconds": round(item.end_seconds, 6),
                "label": item.label,
            }
            for item in project.sublabels
        ],
    }


def _source_sublabels_path(project: ProjectState) -> Path | None:
    """Return the dataset-root sidecar path for resumable sublabels.

    When the project is bound to a specific LeRobot episode, use an
    episode-specific file name (`sublabels_episode_NNNN.json`) so each episode
    keeps its labels separate from the others.
    """
    if project.source_dataset_root is None:
        return None
    root = project.source_dataset_root.expanduser().resolve()

    if project.episode_index is not None:
        return root / f"sublabels_episode_{int(project.episode_index):04d}.json"

    return root / SUBLABELS_FILE_NAME


def _resolve_segments_from_inputs(
    provided_paths: list[Path] | None,
    *,
    stored_payload: Any,
    base_dir: Path,
    fallback: Path | None,
) -> list[Path]:
    """Resolve segment paths from explicit inputs, workspace metadata, or fallback path."""
    if provided_paths:
        return [path.expanduser().resolve() for path in provided_paths]
    if isinstance(stored_payload, list) and stored_payload:
        resolved_paths: list[Path] = []
        for raw_path in stored_payload:
            if not raw_path:
                continue
            resolved_paths.append((base_dir / str(raw_path)).expanduser().resolve())
        if resolved_paths:
            return resolved_paths
    if fallback is None:
        return []
    return [fallback.expanduser().resolve()]


def _load_labels(
    meta_payload: dict[str, Any],
    annotations_payload: Any,
    annotations: dict[int, str],
    sublabels: list[TimeRangeLabel],
) -> list[str]:
    """Resolve the label list while preserving legacy compatibility."""
    raw_labels = meta_payload.get("labels")
    if isinstance(raw_labels, list):
        labels = _normalize_labels(raw_labels)
        if labels:
            return labels

    if isinstance(annotations_payload, list):
        labels_from_annotations: list[str] = []
        seen_labels: set[str] = set()
        for item in annotations_payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label or label in seen_labels:
                continue
            labels_from_annotations.append(label)
            seen_labels.add(label)
        if labels_from_annotations:
            return labels_from_annotations

    if annotations:
        return _normalize_labels(annotations.values())
    if sublabels:
        return _normalize_labels(item.label for item in sublabels)
    return list(ProjectState().labels)


def _normalize_labels(labels: Any) -> list[str]:
    """Normalize user labels while preserving order."""
    normalized_labels: list[str] = []
    seen_labels: set[str] = set()
    for label in labels:
        normalized_label = str(label).strip()
        if not normalized_label or normalized_label in seen_labels:
            continue
        normalized_labels.append(normalized_label)
        seen_labels.add(normalized_label)
    return normalized_labels


def _resolve_current_frame_index(
    meta_payload: dict[str, Any],
    annotations: dict[int, str],
) -> int:
    """Resolve the frame pointer used when the workspace is reopened."""
    raw_current_frame_index = meta_payload.get("current_frame_index")
    if raw_current_frame_index is not None:
        try:
            return max(0, int(raw_current_frame_index))
        except (TypeError, ValueError):
            pass

    raw_last_labeled_frame_index = meta_payload.get("last_labeled_frame_index")
    if raw_last_labeled_frame_index is not None:
        try:
            return max(0, int(raw_last_labeled_frame_index))
        except (TypeError, ValueError):
            pass

    if annotations:
        return max(annotations)
    return 0


def _write_json_atomic(target_path: Path, payload: Any) -> None:
    """Atomically replace one JSON file to reduce autosave corruption risk."""
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
