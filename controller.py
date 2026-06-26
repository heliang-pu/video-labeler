"""Controller layer that keeps UI code separate from labeling operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .exporter import ExportResult, sanitize_dataset_name
from .label_groups import LabelGroupRepository, normalize_label_list
from .models import (
    DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
    DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE,
    ProjectState,
    TimeRangeLabel,
)
from .storage import load_project, save_project
from .time_ranges import frame_indices_for_seconds_range
from .video import VideoFrameReader
from .workspace_store import (
    WorkspaceLoadResult,
    load_project_from_dataset_dir,
    persist_project_workspace,
    workspace_has_state,
)


@dataclass(slots=True)
class RangeLabelResult:
    """Summary of a batch range-labeling operation."""

    start_index: int
    end_index: int
    last_frame_index: int
    step_used: int
    visited_frames: int
    updated_frames: int


@dataclass(slots=True)
class ApplyLabelGroupResult:
    """Summary of applying one reusable label group to the active project."""

    group_name: str
    stored_label_count: int
    final_label_count: int
    preserved_annotation_labels: list[str]


@dataclass(slots=True)
class SaveLabelGroupResult:
    """Summary of saving current project labels as a reusable label group."""

    group_name: str
    label_count: int
    overwritten: bool


class LabelingController:
    """Application service for video loading, labeling, persistence, and export."""

    def __init__(
        self,
        *,
        reader: VideoFrameReader | None = None,
        project: ProjectState | None = None,
        label_group_repository: LabelGroupRepository | None = None,
    ) -> None:
        self._reader = reader or VideoFrameReader()
        self._project = project or ProjectState()
        self._label_group_repository = label_group_repository or LabelGroupRepository()
        self._label_counts: Counter[str] = Counter(self._project.annotations.values())

    @property
    def project(self) -> ProjectState:
        """Expose the current project state."""
        return self._project

    @property
    def reader(self) -> VideoFrameReader:
        """Expose the current video reader."""
        return self._reader

    @property
    def frame_count(self) -> int:
        """Return the frame count of the currently loaded video."""
        return self._reader.frame_count

    @property
    def frame_step(self) -> int:
        """Return the current navigation / labeling step size."""
        return max(1, int(self._project.frame_step))

    @property
    def keyframe_threshold(self) -> float:
        """Return the current threshold used by keyframe navigation."""
        return max(0.0, float(self._project.keyframe_threshold))

    @property
    def active_label_group_name(self) -> str | None:
        """Return the label-group name currently associated with the project."""
        return self._project.label_group_name

    @property
    def label_group_store_path(self) -> Path:
        """Return the global label-group store path."""
        return self._label_group_repository.storage_path

    def label_count(self, label: str) -> int:
        """Return how many saved annotations currently use the label."""
        frame_label_count = int(self._label_counts.get(label, 0))
        range_label_count = sum(1 for item in self._project.sublabels if item.label == label)
        return frame_label_count + range_label_count

    def list_label_groups(self) -> list[str]:
        """Return all persisted label-group names."""
        return self._label_group_repository.list_group_names()

    def label_group_labels(self, group_name: str) -> list[str] | None:
        """Return the labels stored for one label group."""
        return self._label_group_repository.get_group_labels(group_name)

    def replace_project(self, project: ProjectState) -> None:
        """Replace the active project while keeping the existing reader object."""
        self._project = project
        self._rebuild_label_counts()

    def close_video(self) -> None:
        """Release the active decoder without deleting project data."""
        self._reader.close()

    def load_videos(
        self,
        video_path: Path,
        wrist_video_path: Path | None = None,
        left_wrist_video_path: Path | None = None,
        *,
        video_segment_paths: list[Path] | None = None,
        wrist_video_segment_paths: list[Path] | None = None,
        left_wrist_video_segment_paths: list[Path] | None = None,
        source_dataset_root: Path | None = None,
        preserve_project: bool = False,
    ) -> None:
        """Open one, two, or three synchronized video files and initialize project state."""
        resolved_video_segments = self._resolve_required_segments(video_path, video_segment_paths)
        resolved_wrist_segments = self._resolve_optional_segments(
            wrist_video_path,
            wrist_video_segment_paths,
        )
        resolved_left_wrist_segments = self._resolve_optional_segments(
            left_wrist_video_path,
            left_wrist_video_segment_paths,
        )
        resolved_video_path = resolved_video_segments[0]
        resolved_wrist_video_path = resolved_wrist_segments[0] if resolved_wrist_segments else None
        resolved_left_wrist_video_path = (
            resolved_left_wrist_segments[0] if resolved_left_wrist_segments else None
        )
        resolved_source_dataset_root = (
            None if source_dataset_root is None else source_dataset_root.expanduser().resolve()
        )

        self._reader.open(
            resolved_video_path,
            resolved_wrist_video_path,
            resolved_left_wrist_video_path,
            video_segment_paths=resolved_video_segments,
            wrist_video_segment_paths=resolved_wrist_segments,
            left_wrist_video_segment_paths=resolved_left_wrist_segments,
        )

        if preserve_project:
            self._project.video_path = resolved_video_path
            self._project.wrist_video_path = resolved_wrist_video_path
            self._project.left_wrist_video_path = resolved_left_wrist_video_path
            self._project.video_segment_paths = list(resolved_video_segments)
            self._project.wrist_video_segment_paths = list(resolved_wrist_segments)
            self._project.left_wrist_video_segment_paths = list(resolved_left_wrist_segments)
            self._project.source_dataset_root = resolved_source_dataset_root
            if not self._project.dataset_name.strip():
                self._project.dataset_name = sanitize_dataset_name(resolved_video_path.stem)
            # Auto-select appropriate input template based on number of views
            if self._project.input_template.strip() == ProjectState().input_template:
                if resolved_left_wrist_video_path is not None:
                    self._project.input_template = DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE
                elif resolved_wrist_video_path is not None:
                    self._project.input_template = DEFAULT_DUAL_VIEW_INPUT_TEMPLATE
        else:
            new_project = ProjectState(
                video_path=resolved_video_path,
                wrist_video_path=resolved_wrist_video_path,
                left_wrist_video_path=resolved_left_wrist_video_path,
                video_segment_paths=list(resolved_video_segments),
                wrist_video_segment_paths=list(resolved_wrist_segments),
                left_wrist_video_segment_paths=list(resolved_left_wrist_segments),
                source_dataset_root=resolved_source_dataset_root,
            )
            new_project.dataset_name = sanitize_dataset_name(resolved_video_path.stem)
            if resolved_left_wrist_video_path is not None:
                new_project.input_template = DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE
            elif resolved_wrist_video_path is not None:
                new_project.input_template = DEFAULT_DUAL_VIEW_INPUT_TEMPLATE
            self._project = new_project
            self._rebuild_label_counts()

        if self._reader.frame_count > 0:
            self._project.current_frame_index = min(
                max(0, self._project.current_frame_index),
                self._reader.frame_count - 1,
            )
        else:
            self._project.current_frame_index = 0

    def load_video(self, video_path: Path, *, preserve_project: bool = False) -> None:
        """Backward-compatible wrapper for loading only the top view."""
        self.load_videos(video_path, None, preserve_project=preserve_project)

    def read_frame(self, frame_index: int):
        """Read one frame and update the current frame pointer."""
        if self._reader.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")
        clamped_index = max(0, min(int(frame_index), self._reader.frame_count - 1))
        frame = self._reader.read_frame_bgr(clamped_index)
        self._project.current_frame_index = clamped_index
        return frame

    def read_frames(self, frame_index: int) -> dict[str, object]:
        """Read synchronized frames from all active views."""
        if self._reader.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")
        clamped_index = max(0, min(int(frame_index), self._reader.frame_count - 1))
        frames = self._reader.read_frames_bgr(clamped_index)
        self._project.current_frame_index = clamped_index
        return frames

    def set_frame_step(self, frame_step: int) -> None:
        """Update the frame stepping interval used by browser navigation."""
        self._project.frame_step = max(1, int(frame_step))

    def set_keyframe_threshold(self, threshold: float) -> None:
        """Update the threshold used when searching for the next distinct frame."""
        self._project.keyframe_threshold = max(0.0, float(threshold))

    def step_forward_index(self) -> int:
        """Return the next frame index according to the configured step size."""
        return self._bounded_frame_index(self._project.current_frame_index + self.frame_step)

    def step_backward_index(self) -> int:
        """Return the previous frame index according to the configured step size."""
        return self._bounded_frame_index(self._project.current_frame_index - self.frame_step)

    def add_label(self, label: str) -> None:
        """Append a new label if it does not already exist."""
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("Label cannot be empty.")
        if normalized_label in self._project.labels:
            raise ValueError(f"Label already exists: {normalized_label}")
        self._project.labels.append(normalized_label)
        self._project.label_group_name = None

    def apply_label_group(self, group_name: str) -> ApplyLabelGroupResult:
        """Replace the current label list with one stored reusable group."""
        stored_labels = self._label_group_repository.get_group_labels(group_name)
        if stored_labels is None:
            raise ValueError(f"标签组不存在: {group_name.strip()}")

        preserved_annotation_labels = self._annotation_labels_not_in(stored_labels)
        self._project.labels = list(stored_labels) + preserved_annotation_labels
        self._project.label_group_name = group_name.strip()
        return ApplyLabelGroupResult(
            group_name=self._project.label_group_name,
            stored_label_count=len(stored_labels),
            final_label_count=len(self._project.labels),
            preserved_annotation_labels=preserved_annotation_labels,
        )

    def save_current_labels_as_group(
        self,
        group_name: str,
        *,
        overwrite: bool = False,
    ) -> SaveLabelGroupResult:
        """Persist the current project labels as one reusable label group."""
        normalized_project_labels = normalize_label_list(self._project.labels)
        normalized_group_name, overwritten = self._label_group_repository.save_group(
            group_name,
            normalized_project_labels,
            overwrite=overwrite,
        )
        self._project.label_group_name = normalized_group_name
        return SaveLabelGroupResult(
            group_name=normalized_group_name,
            label_count=len(normalized_project_labels),
            overwritten=overwritten,
        )

    def delete_label_group(self, group_name: str) -> None:
        """Delete one reusable label group from the global store."""
        normalized_group_name = group_name.strip()
        self._label_group_repository.delete_group(normalized_group_name)
        if self._project.label_group_name == normalized_group_name:
            self._project.label_group_name = None

    def assign_label(self, label: str) -> None:
        """Assign a label to the current frame and update incremental counts."""
        self.assign_label_at(self._project.current_frame_index, label)

    def assign_label_at(self, frame_index: int, label: str) -> bool:
        """Assign a label to one frame and return whether anything changed."""
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("Label cannot be empty.")
        if normalized_label not in self._project.labels:
            raise ValueError(f"Unknown label: {normalized_label}")

        bounded_index = self._bounded_frame_index(frame_index)
        previous_label = self._project.annotation_for(bounded_index)
        if previous_label == normalized_label:
            return False
        if previous_label is not None:
            self._label_counts[previous_label] -= 1
        self._project.set_annotation(bounded_index, normalized_label)
        self._label_counts[normalized_label] += 1
        return True

    def clear_current_annotation(self) -> None:
        """Remove the label from the current frame if present."""
        frame_index = self._project.current_frame_index
        previous_label = self._project.annotation_for(frame_index)
        if previous_label is None:
            return
        self._project.clear_annotation(frame_index)
        self._label_counts[previous_label] -= 1

    def save_project_file(self, project_path: Path) -> Path:
        """Persist the current project to disk."""
        save_project(project_path, self._project)
        self._project.project_path = project_path.expanduser().resolve()
        return self._project.project_path

    def load_project_file(self, project_path: Path) -> ProjectState:
        """Load a saved project from disk."""
        project = load_project(project_path)
        self.replace_project(project)
        return project

    def dataset_workspace_exists(self, output_dir: Path) -> bool:
        """Return whether a dataset workspace already exists on disk."""
        return workspace_has_state(output_dir)

    def load_dataset_workspace(
        self,
        output_dir: Path,
        *,
        video_path: Path | None = None,
        wrist_video_path: Path | None = None,
        left_wrist_video_path: Path | None = None,
        video_segment_paths: list[Path] | None = None,
        wrist_video_segment_paths: list[Path] | None = None,
        left_wrist_video_segment_paths: list[Path] | None = None,
        source_dataset_root: Path | None = None,
    ) -> WorkspaceLoadResult:
        """Load project state from an existing dataset workspace directory."""
        result = load_project_from_dataset_dir(
            output_dir,
            video_path=video_path,
            wrist_video_path=wrist_video_path,
            left_wrist_video_path=left_wrist_video_path,
            video_segment_paths=video_segment_paths,
            wrist_video_segment_paths=wrist_video_segment_paths,
            left_wrist_video_segment_paths=left_wrist_video_segment_paths,
            source_dataset_root=source_dataset_root,
        )
        self.replace_project(result.project)
        return result

    def save_dataset_workspace(self, output_dir: Path) -> ExportResult:
        """Persist the active project into the dataset workspace directory."""
        return persist_project_workspace(
            self._project,
            output_dir,
            reader=self._reader,
            require_annotations=False,
        )

    def update_export_settings(self, *, dataset_name: str, instruction_template: str, input_template: str) -> None:
        """Update the project settings used during dataset export."""
        default_input_template = ProjectState().input_template
        if self._project.has_dual_views():
            default_input_template = DEFAULT_DUAL_VIEW_INPUT_TEMPLATE
        self._project.dataset_name = dataset_name.strip() or "video_labels"
        self._project.instruction_template = (
            instruction_template.strip() or ProjectState().instruction_template
        )
        self._project.input_template = input_template.strip() or default_input_template

    def validate_export_settings(self) -> None:
        """Ensure the current export templates are compatible with the loaded view count."""
        image_token_count = self._project.input_template.count("<image>")
        required_image_token_count = max(1, self._project.active_view_count())
        if image_token_count != required_image_token_count:
            raise ValueError(
                "当前项目会为每条样本导出 "
                f"{required_image_token_count} 张图片，"
                f"所以 input 模板中必须恰好包含 {required_image_token_count} 个 <image>。"
            )

    def export_dataset(self, output_dir: Path, progress_callback=None) -> ExportResult:
        """Validate settings and export a LLaMA-Factory dataset."""
        self.validate_export_settings()
        return persist_project_workspace(
            self._project,
            output_dir,
            reader=self._reader,
            require_annotations=True,
            progress_callback=progress_callback,
        )

    def find_next_unlabeled_index(self, *, use_frame_step: bool = True) -> int | None:
        """Return the next unlabeled frame index after the current frame."""
        if self._reader.frame_count <= 0:
            return None

        for frame_index in self._iter_future_indices(use_frame_step=use_frame_step):
            if self._project.annotation_for(frame_index) is None:
                return frame_index
        return None

    def apply_label_to_range(
        self,
        start_frame_index: int,
        end_frame_index: int,
        label: str,
        *,
        use_frame_step: bool = True,
    ) -> RangeLabelResult:
        """Apply one label to an inclusive frame interval."""
        if self._reader.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")

        step = self.frame_step if use_frame_step else 1
        start_index = self._bounded_frame_index(start_frame_index)
        end_index = self._bounded_frame_index(end_frame_index)
        if start_index > end_index:
            start_index, end_index = end_index, start_index

        updated_frames = 0
        visited_frames = 0
        last_frame_index = start_index

        for frame_index in range(start_index, end_index + 1, step):
            visited_frames += 1
            last_frame_index = frame_index
            if self.assign_label_at(frame_index, label):
                updated_frames += 1

        self._project.current_frame_index = last_frame_index
        return RangeLabelResult(
            start_index=start_index,
            end_index=end_index,
            last_frame_index=last_frame_index,
            step_used=step,
            visited_frames=visited_frames,
            updated_frames=updated_frames,
        )

    def add_sublabel_seconds(self, start_seconds: float, end_seconds: float, label: str) -> TimeRangeLabel:
        """Add one seconds-based interval label to the project and annotate all frames in the range."""
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("Label cannot be empty.")
        if normalized_label not in self._project.labels:
            raise ValueError(f"Unknown label: {normalized_label}")
        if self._reader.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")

        if self._reader.fps > 0:
            total_seconds = self._reader.frame_count / self._reader.fps
        else:
            total_seconds = self._reader.timestamp_seconds(max(0, self._reader.frame_count - 1))

        bounded_start = max(0.0, float(start_seconds))
        bounded_end = max(0.0, float(end_seconds))
        if bounded_start > bounded_end:
            bounded_start, bounded_end = bounded_end, bounded_start
        bounded_start = min(bounded_start, total_seconds)
        bounded_end = min(bounded_end, total_seconds)
        if bounded_end <= bounded_start:
            raise ValueError("结束时间必须大于开始时间。")

        # Add sublabel for time range record
        sublabel = TimeRangeLabel(
            start_seconds=bounded_start,
            end_seconds=bounded_end,
            label=normalized_label,
        )
        self._project.sublabels.append(sublabel)
        self._project.sublabels.sort(
            key=lambda item: (item.start_seconds, item.end_seconds, item.label)
        )

        covered_frames = frame_indices_for_seconds_range(
            bounded_start,
            bounded_end,
            fps=self._reader.fps,
            frame_count=self._reader.frame_count,
        )
        for frame_index in covered_frames:
            self._project.annotations[frame_index] = normalized_label
            if normalized_label in self._label_counts:
                self._label_counts[normalized_label] += 1
            else:
                self._label_counts[normalized_label] = 1

        if covered_frames:
            self._project.current_frame_index = self._bounded_frame_index(covered_frames[0])
        return sublabel

    def delete_sublabel(self, row_index: int) -> TimeRangeLabel:
        """Delete a time-range label by table row index."""
        if not self._project.sublabels:
            raise ValueError("当前没有可删除的区间标签。")
        index = int(row_index)
        if index < 0 or index >= len(self._project.sublabels):
            raise ValueError(f"区间序号超出范围: {index}")
        return self._project.sublabels.pop(index)

    def find_next_distinct_frame(
        self,
        *,
        threshold: float | None = None,
        use_frame_step: bool = True,
    ) -> int | None:
        """Return the next frame whose signature differs enough from the current frame."""
        if self._reader.frame_count <= 0:
            return None

        reference_index = self._project.current_frame_index
        effective_threshold = self.keyframe_threshold if threshold is None else max(0.0, float(threshold))
        for frame_index in self._iter_future_indices(use_frame_step=use_frame_step):
            if self._reader.frame_difference_score(reference_index, frame_index) >= effective_threshold:
                return frame_index
        return None

    def _rebuild_label_counts(self) -> None:
        """Recompute label counts from scratch, e.g. after loading a project."""
        self._label_counts = Counter(self._project.annotations.values())

    def _annotation_labels_not_in(self, labels: list[str]) -> list[str]:
        """Return labels already used by annotations but absent from the given label list."""
        normalized_labels = set(labels)
        preserved_labels: list[str] = []
        used_labels = [*self._project.annotations.values(), *(item.label for item in self._project.sublabels)]
        for label in used_labels:
            if label in normalized_labels or label in preserved_labels:
                continue
            preserved_labels.append(label)
        return preserved_labels

    def _iter_future_indices(self, *, use_frame_step: bool = True):
        """Yield future frame indices starting after the current one."""
        if self._reader.frame_count <= 0:
            return

        step = self.frame_step if use_frame_step else 1
        next_index = self._project.current_frame_index + step
        while next_index < self._reader.frame_count:
            yield next_index
            next_index += step

    def _bounded_frame_index(self, frame_index: int) -> int:
        """Clamp a raw index into the currently loaded video range."""
        if self._reader.frame_count <= 0:
            return 0
        return max(0, min(int(frame_index), self._reader.frame_count - 1))

    @staticmethod
    def _resolve_required_segments(
        video_path: Path,
        video_segment_paths: list[Path] | None,
    ) -> list[Path]:
        """Resolve a required video segment list."""
        raw_paths = video_segment_paths if video_segment_paths else [video_path]
        return [path.expanduser().resolve() for path in raw_paths]

    @staticmethod
    def _resolve_optional_segments(
        video_path: Path | None,
        video_segment_paths: list[Path] | None,
    ) -> list[Path]:
        """Resolve an optional video segment list."""
        if video_segment_paths:
            return [path.expanduser().resolve() for path in video_segment_paths]
        if video_path is None:
            return []
        return [video_path.expanduser().resolve()]
