"""Data models shared by the video labeler and dataset exporter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_LABELS = ["left_hand_pickup", "object_at_center", "right_hand_transfer"]
DEFAULT_LABEL_GROUP_NAME = "parcel_sorting_bimanual"
DEFAULT_SINGLE_VIEW_INPUT_TEMPLATE = "<image>"
DEFAULT_DUAL_VIEW_INPUT_TEMPLATE = "top视角: <image>\nwrist视角: <image>"
DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE = "top视角: <image>\nright_wrist视角: <image>\nleft_wrist视角: <image>"


@dataclass(slots=True)
class TimeRangeLabel:
    """One user label assigned to a continuous time interval."""

    start_seconds: float
    end_seconds: float
    label: str

    def normalized(self) -> "TimeRangeLabel":
        """Return the interval with ordered, non-negative bounds."""
        start_seconds = max(0.0, float(self.start_seconds))
        end_seconds = max(0.0, float(self.end_seconds))
        if start_seconds > end_seconds:
            start_seconds, end_seconds = end_seconds, start_seconds
        return TimeRangeLabel(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            label=self.label.strip(),
        )


@dataclass(slots=True)
class ProjectState:
    """User-editable labeling project state."""

    video_path: Path | None = None
    wrist_video_path: Path | None = None
    left_wrist_video_path: Path | None = None
    video_segment_paths: list[Path] = field(default_factory=list)
    wrist_video_segment_paths: list[Path] = field(default_factory=list)
    left_wrist_video_segment_paths: list[Path] = field(default_factory=list)
    source_dataset_root: Path | None = None
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    annotations: dict[int, str] = field(default_factory=dict)
    sublabels: list[TimeRangeLabel] = field(default_factory=list)
    current_frame_index: int = 0
    frame_step: int = 1
    keyframe_threshold: float = 12.0
    label_group_name: str | None = DEFAULT_LABEL_GROUP_NAME
    instruction_template: str = """Based on three camera views (top, right_wrist, left_wrist), determine the current phase of the robot task.

Task: Parcel Sorting (Bimanual Coordination)
- left_hand_pickup: Left hand picks an object from the pile and places it at the center area
- object_at_center: Object is placed at center area, ready for right hand pickup
- right_hand_transfer: Right hand picks from center and transfers to the right side

Output only the phase label name."""
    input_template: str = DEFAULT_SINGLE_VIEW_INPUT_TEMPLATE
    dataset_name: str = "video_labels"
    project_path: Path | None = None
    episode_index: int | None = None

    def set_annotation(self, frame_index: int, label: str) -> None:
        """Set or replace the label assigned to one frame."""
        self.annotations[int(frame_index)] = label

    def clear_annotation(self, frame_index: int) -> None:
        """Remove the current frame annotation if it exists."""
        self.annotations.pop(int(frame_index), None)

    def annotation_for(self, frame_index: int) -> str | None:
        """Return the assigned label for the frame, if any."""
        return self.annotations.get(int(frame_index))

    def labeled_frame_count(self) -> int:
        """Return how many frames are currently annotated."""
        return len(self.annotations)

    def sublabel_count(self) -> int:
        """Return how many time intervals are currently annotated."""
        return len(self.sublabels)

    def active_video_segments(self) -> dict[str, list[Path]]:
        """Return active video segment lists in deterministic view order."""
        items: dict[str, list[Path]] = {}
        top_segments = list(self.video_segment_paths)
        if not top_segments and self.video_path is not None:
            top_segments = [self.video_path]
        if top_segments:
            items["top"] = top_segments

        wrist_segments = list(self.wrist_video_segment_paths)
        if not wrist_segments and self.wrist_video_path is not None:
            wrist_segments = [self.wrist_video_path]
        if wrist_segments:
            items["wrist"] = wrist_segments

        left_wrist_segments = list(self.left_wrist_video_segment_paths)
        if not left_wrist_segments and self.left_wrist_video_path is not None:
            left_wrist_segments = [self.left_wrist_video_path]
        if left_wrist_segments:
            items["left_wrist"] = left_wrist_segments
        return items

    def active_video_items(self) -> list[tuple[str, Path]]:
        """Return active video views in deterministic export order."""
        items: list[tuple[str, Path]] = []
        for view_name, segment_paths in self.active_video_segments().items():
            if segment_paths:
                items.append((view_name, segment_paths[0]))
        return items

    def active_view_count(self) -> int:
        """Return how many video views are currently loaded."""
        return len(self.active_video_items())

    def has_dual_views(self) -> bool:
        """Return whether both top and wrist views are configured."""
        return self.video_path is not None and self.wrist_video_path is not None

    def has_triple_views(self) -> bool:
        """Return whether all three views (top, wrist, left_wrist) are configured."""
        return (
            self.video_path is not None
            and self.wrist_video_path is not None
            and self.left_wrist_video_path is not None
        )
