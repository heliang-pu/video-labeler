"""Stateful Gradio adapter built on top of the shared labeling controller."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2

from .controller import LabelingController
from .episode_metadata import (
    EpisodeCatalog,
    EpisodeRange,
    load_episode_catalog,
    resolve_view_video_path,
)
from .exporter import sanitize_dataset_name
from .label_groups import LabelGroupRepository
from .models import (
    DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
    DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE,
    ProjectState,
)
from .time_ranges import frame_indices_for_seconds_range
from .video_sources import resolve_video_inputs
from .workspace_store import SUBLABELS_FILE_NAME


DATASET_ROOT_DIR = Path("/workspace/dataset")
LEROBOT_DATASET_DIR = DATASET_ROOT_DIR / "lerobot"
QWEN_FINETUNE_DATASET_DIR = DATASET_ROOT_DIR / "qwen_finetune"
DEFAULT_EXPORT_DIR = QWEN_FINETUNE_DATASET_DIR
MAX_PREVIEW_EDGE = 1280


def suggest_export_dir_for_video(
    resolved_top_video_path: Path,
    *,
    lerobot_dataset_name: str = "",
    selected_relative_path: str = "",
    base_dir: Path = DEFAULT_EXPORT_DIR,
) -> Path:
    """Build one stable default dataset directory for the current video pair."""
    dataset_key_source = lerobot_dataset_name.strip()
    if not dataset_key_source:
        dataset_key_source = selected_relative_path.strip()
    if dataset_key_source and not lerobot_dataset_name.strip():
        dataset_key_source = str(Path(dataset_key_source).with_suffix(""))
    if not dataset_key_source:
        dataset_key_source = resolved_top_video_path.stem
    dataset_key = sanitize_dataset_name(dataset_key_source)
    return base_dir.expanduser().resolve() / dataset_key


@dataclass(slots=True)
class ViewSnapshot:
    """UI-facing snapshot used by the Gradio frontend."""

    top_frame_image: Any | None
    wrist_frame_image: Any | None
    left_wrist_frame_image: Any | None
    frame_index: int
    frame_count: int
    current_seconds: float
    frame_info_markdown: str
    project_markdown: str
    label_summary_markdown: str
    sublabels_table: list[list[object]]
    labels: list[str]
    label_group_names: list[str]
    selected_label: str | None
    range_label: str | None
    selected_label_group: str | None
    active_label_group_name: str | None
    status_message: str
    top_input_path: str
    wrist_input_path: str
    selected_relative_path: str
    available_relative_paths: list[str]
    resolved_top_video_path: str
    resolved_wrist_video_path: str
    resolved_left_wrist_video_path: str
    dataset_name: str
    instruction_template: str
    input_template: str
    project_path: str
    export_dir: str
    frame_step: int
    keyframe_threshold: float
    label_group_store_path: str
    fps: float
    total_seconds: float
    timeline_image: Any | None
    sync_player_html: str


class GradioLabelingSession:
    """Per-browser-session state container for the Gradio UI."""

    def __init__(self) -> None:
        self.controller = LabelingController()
        self.project_path_text = ""
        self.export_dir_text = str(DEFAULT_EXPORT_DIR)
        self.export_dir_auto_managed = True
        self.lerobot_dataset_name = ""
        self.status_message = "准备就绪。请输入 top 视频路径后点击“加载视频”。"
        self.range_label_selection: str | None = None
        self.selected_label_group_name: str | None = None
        self.top_input_path_text = ""
        self.wrist_input_path_text = ""
        self.selected_relative_path_text = ""
        self.available_relative_paths: list[str] = []
        self._frame_image_cache: dict[str, Any] = {}
        self._frame_image_cache_index = -1
        self._frame_image_cache_signature: tuple[str | None, str | None, str | None] | None = None
        self.episode_catalog: EpisodeCatalog | None = None
        self.current_episode_index: int | None = None

    def __deepcopy__(self, memo):
        """Rebuild one equivalent session without sharing live OpenCV handles."""
        copied_session = type(self)()
        copied_session.controller = LabelingController(
            project=deepcopy(self.controller.project, memo),
            label_group_repository=LabelGroupRepository(self.controller.label_group_store_path),
        )
        if copied_session.controller.project.video_path is not None:
            copied_session.controller.load_videos(
                copied_session.controller.project.video_path,
                copied_session.controller.project.wrist_video_path,
                copied_session.controller.project.left_wrist_video_path,
                video_segment_paths=copied_session.controller.project.video_segment_paths,
                wrist_video_segment_paths=copied_session.controller.project.wrist_video_segment_paths,
                left_wrist_video_segment_paths=(
                    copied_session.controller.project.left_wrist_video_segment_paths
                ),
                source_dataset_root=copied_session.controller.project.source_dataset_root,
                preserve_project=True,
            )
            # Re-apply episode windowing if the project is bound to one episode.
            project = copied_session.controller.project
            if (
                project.episode_index is not None
                and project.source_dataset_root is not None
            ):
                catalog = load_episode_catalog(project.source_dataset_root)
                if catalog is not None:
                    episode = catalog.get(int(project.episode_index))
                    if episode is not None:
                        copied_session.episode_catalog = catalog
                        copied_session.current_episode_index = int(project.episode_index)
                        fps = copied_session.controller.reader.fps
                        offsets: dict[str, int] = {}
                        if fps > 0:
                            for view_name, window in episode.views.items():
                                offsets[view_name] = int(round(window.from_seconds * fps))
                        copied_session.controller.reader.set_episode_window(
                            offsets, episode.length_frames
                        )
        copied_session.project_path_text = self.project_path_text
        copied_session.export_dir_text = self.export_dir_text
        copied_session.export_dir_auto_managed = self.export_dir_auto_managed
        copied_session.lerobot_dataset_name = self.lerobot_dataset_name
        copied_session.status_message = self.status_message
        copied_session.range_label_selection = self.range_label_selection
        copied_session.selected_label_group_name = self.selected_label_group_name
        copied_session.top_input_path_text = self.top_input_path_text
        copied_session.wrist_input_path_text = self.wrist_input_path_text
        copied_session.selected_relative_path_text = self.selected_relative_path_text
        copied_session.available_relative_paths = list(self.available_relative_paths)
        copied_session.episode_catalog = copied_session.episode_catalog or self.episode_catalog
        copied_session.current_episode_index = copied_session.current_episode_index or self.current_episode_index
        copied_session._invalidate_frame_cache()
        return copied_session

    def sync_export_settings(
        self,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
    ) -> "GradioLabelingSession":
        """Persist editable export text fields into the shared controller."""
        self.controller.update_export_settings(
            dataset_name=dataset_name,
            instruction_template=instruction_template,
            input_template=input_template,
        )
        return self

    def set_frame_step(self, frame_step: int | float) -> "GradioLabelingSession":
        """Update the frame stepping interval used by navigation and auto-advance."""
        self.controller.set_frame_step(int(frame_step))
        self.status_message = self._with_autosave_status(
            f"当前标注步长已设置为每 {self.controller.frame_step} 帧。"
        )
        return self

    def set_keyframe_threshold(self, threshold: int | float) -> "GradioLabelingSession":
        """Update the threshold used by the next-keyframe navigation button."""
        self.controller.set_keyframe_threshold(float(threshold))
        self.status_message = self._with_autosave_status(
            f"关键帧阈值已设置为 {self.controller.keyframe_threshold:.2f}，"
            "数值越小越容易判定为新关键帧。"
        )
        return self

    def set_range_label(self, label: str | None) -> "GradioLabelingSession":
        """Remember the label currently selected for range labeling."""
        self.range_label_selection = self._ensure_range_label_selection(preferred=label)
        return self

    def set_selected_label_group(self, group_name: str | None) -> "GradioLabelingSession":
        """Remember which reusable label group is selected in the UI."""
        self.selected_label_group_name = self._ensure_selected_label_group(preferred=group_name)
        return self

    def load_episode(
        self,
        dataset_root_text: str,
        episode_index: int,
        export_dir_text: str,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
    ) -> "GradioLabelingSession":
        """Load one specific episode from a LeRobot dataset using `meta/episodes/`."""
        normalized_root_text = (dataset_root_text or "").strip()
        if not normalized_root_text:
            self.status_message = "请输入 LeRobot 数据集目录。"
            return self
        try:
            dataset_root = Path(normalized_root_text).expanduser().resolve()
        except Exception as exc:
            self.status_message = f"数据集目录无效: {exc}"
            return self
        if not dataset_root.is_dir():
            self.status_message = f"数据集目录不存在: {dataset_root}"
            return self

        catalog = load_episode_catalog(dataset_root)
        if catalog is None:
            self.status_message = (
                f"未在数据集中找到 `meta/episodes/`，无法按 episode 加载。请确认数据集是 LeRobot v3 格式: {dataset_root}"
            )
            return self
        self.episode_catalog = catalog

        episode = catalog.get(int(episode_index))
        if episode is None:
            available = sorted(catalog.episodes.keys())
            sample = ", ".join(str(idx) for idx in available[:5])
            self.status_message = f"未找到 episode {episode_index}。可用范围: 0 - {max(available)} (例如 {sample})。"
            return self

        # Resolve per-view single-file paths from the catalog window.
        view_paths: dict[str, Path] = {}
        for view_name, window in episode.views.items():
            resolved = resolve_view_video_path(
                dataset_root,
                view_name,
                window.chunk_index,
                window.file_index,
            )
            if resolved is not None:
                view_paths[view_name] = resolved

        if "top" not in view_paths:
            self.status_message = (
                f"episode {episode_index} 缺少 top 视频文件: chunk-{episode.views.get('top').chunk_index:03d}/file-{episode.views.get('top').file_index:03d}.mp4"
                if episode.views.get("top") is not None
                else f"episode {episode_index} 缺少 top 视图元数据。"
            )
            return self

        top_path = view_paths["top"]
        wrist_path = view_paths.get("wrist")
        left_path = view_paths.get("left_wrist")

        resolved_export_dir_text, export_dir_auto_managed = self._resolve_export_dir_text(
            export_dir_text,
            top_path,
            lerobot_dataset_name=dataset_root.name,
            selected_relative_path=f"episode_{episode_index:04d}",
        )
        self.export_dir_text = resolved_export_dir_text
        self.export_dir_auto_managed = export_dir_auto_managed
        self.lerobot_dataset_name = dataset_root.name
        self.top_input_path_text = normalized_root_text
        self.wrist_input_path_text = ""
        self.selected_relative_path_text = f"episode_{episode_index:04d}"
        self.available_relative_paths = []
        self.current_episode_index = int(episode_index)

        effective_input_template = (input_template or "").strip() or ProjectState().input_template
        if effective_input_template == ProjectState().input_template:
            if left_path is not None:
                effective_input_template = DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE
            elif wrist_path is not None:
                effective_input_template = DEFAULT_DUAL_VIEW_INPUT_TEMPLATE

        try:
            self.controller.load_videos(
                top_path,
                wrist_path,
                left_path,
                video_segment_paths=[top_path],
                wrist_video_segment_paths=[wrist_path] if wrist_path else None,
                left_wrist_video_segment_paths=[left_path] if left_path else None,
                source_dataset_root=dataset_root,
                preserve_project=False,
            )
        except Exception as exc:
            self.status_message = f"加载 episode {episode_index} 失败: {exc}"
            self._invalidate_frame_cache()
            return self

        # Apply per-view frame offset windowing.
        fps = self.controller.reader.fps
        view_frame_offsets: dict[str, int] = {}
        if fps > 0:
            for view_name, window in episode.views.items():
                if view_name in view_paths:
                    view_frame_offsets[view_name] = int(round(window.from_seconds * fps))
        self.controller.reader.set_episode_window(view_frame_offsets, episode.length_frames)
        # Snap project frame index back to 0 of the new episode window.
        self.controller.project.current_frame_index = 0
        self.controller.project.episode_index = int(episode_index)

        self.sync_export_settings(
            dataset_name,
            instruction_template,
            effective_input_template,
        )

        # Restore per-episode sublabels if present.
        restored_label_count = 0
        sublabels_path = self._episode_sublabels_path(dataset_root, episode_index)
        if sublabels_path is not None and sublabels_path.exists():
            try:
                self._load_sublabels_from_path(sublabels_path)
                restored_label_count = self.controller.project.sublabel_count()
            except Exception as exc:
                self.status_message = (
                    f"episode {episode_index} 已加载，但读取已保存区间标签失败: {exc}"
                )

        self.project_path_text = self._default_project_path().as_posix()
        loaded_views = ["top"]
        if wrist_path is not None:
            loaded_views.append("wrist")
        if left_path is not None:
            loaded_views.append("left_wrist")
        loaded_views_text = " + ".join(loaded_views)
        duration_text = f"{episode.duration_seconds:.2f}s ({episode.length_frames} 帧)"
        self.status_message = (
            f"已加载 episode {episode_index} / {catalog.total_episodes}。"
            f" 视角: {loaded_views_text}。 时长: {duration_text}。"
            f" 已恢复 {restored_label_count} 个区间标签。"
        )
        self._ensure_selected_label_group()
        self._ensure_range_label_selection()
        self._refresh_cached_preview()
        return self

    def _episode_sublabels_path(self, dataset_root: Path, episode_index: int) -> Path | None:
        """Return the path of the sidecar JSON for one episode's sublabels."""
        return dataset_root / f"sublabels_episode_{episode_index:04d}.json"

    def _load_sublabels_from_path(self, path: Path) -> None:
        """Load saved sublabels from a sidecar JSON into the current project."""
        from .workspace_store import _load_sublabels

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.controller.project.sublabels = _load_sublabels(payload)

    def load_video(
        self,
        top_video_path_text: str,
        wrist_video_path_text: str,
        relative_path_text: str,
        export_dir_text: str,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
    ) -> "GradioLabelingSession":
        """Load one or two synchronized video paths entered in the browser UI."""
        try:
            resolved_inputs = resolve_video_inputs(
                top_video_path_text,
                wrist_video_path_text,
                relative_path_text,
            )
        except Exception as exc:
            self.status_message = f"加载视频失败: {exc}"
            self._invalidate_frame_cache()
            return self

        resolved_export_dir_text, export_dir_auto_managed = self._resolve_export_dir_text(
            export_dir_text,
            resolved_inputs.resolved_top_video_path,
            lerobot_dataset_name=resolved_inputs.lerobot_dataset_name,
            selected_relative_path=resolved_inputs.selected_relative_path,
        )
        self.export_dir_text = resolved_export_dir_text
        self.export_dir_auto_managed = export_dir_auto_managed
        self.lerobot_dataset_name = resolved_inputs.lerobot_dataset_name
        self.top_input_path_text = resolved_inputs.top_source_text
        self.wrist_input_path_text = resolved_inputs.wrist_source_text
        self.selected_relative_path_text = resolved_inputs.selected_relative_path
        self.available_relative_paths = list(resolved_inputs.available_relative_paths)
        resolved_top_video_path = resolved_inputs.resolved_top_video_path
        resolved_wrist_video_path = resolved_inputs.resolved_wrist_video_path
        resolved_left_wrist_video_path = resolved_inputs.resolved_left_wrist_video_path
        resolved_top_video_paths = resolved_inputs.resolved_top_video_paths
        resolved_wrist_video_paths = resolved_inputs.resolved_wrist_video_paths
        resolved_left_wrist_video_paths = resolved_inputs.resolved_left_wrist_video_paths
        source_dataset_root = resolved_inputs.source_dataset_root

        effective_input_template = input_template.strip() or ProjectState().input_template
        if effective_input_template == ProjectState().input_template:
            if resolved_left_wrist_video_path is not None:
                effective_input_template = DEFAULT_TRIPLE_VIEW_INPUT_TEMPLATE
            elif resolved_wrist_video_path is not None:
                effective_input_template = DEFAULT_DUAL_VIEW_INPUT_TEMPLATE

        try:
            restore_dir = Path(resolved_export_dir_text)
            source_sublabels_path = None
            if source_dataset_root is not None:
                source_sublabels_path = source_dataset_root / SUBLABELS_FILE_NAME
            if source_sublabels_path is not None and source_sublabels_path.exists():
                restore_dir = source_dataset_root

            if self.controller.dataset_workspace_exists(restore_dir):
                restore_result = self.controller.load_dataset_workspace(
                    restore_dir,
                    video_path=resolved_top_video_path,
                    wrist_video_path=resolved_wrist_video_path,
                    left_wrist_video_path=resolved_left_wrist_video_path,
                    video_segment_paths=resolved_top_video_paths,
                    wrist_video_segment_paths=resolved_wrist_video_paths,
                    left_wrist_video_segment_paths=resolved_left_wrist_video_paths,
                    source_dataset_root=source_dataset_root,
                )
                self.controller.load_videos(
                    restore_result.project.video_path,
                    restore_result.project.wrist_video_path,
                    restore_result.project.left_wrist_video_path,
                    video_segment_paths=restore_result.project.video_segment_paths,
                    wrist_video_segment_paths=restore_result.project.wrist_video_segment_paths,
                    left_wrist_video_segment_paths=restore_result.project.left_wrist_video_segment_paths,
                    source_dataset_root=restore_result.project.source_dataset_root,
                    preserve_project=True,
                )
                status_message = (
                    f"已加载视频并恢复区间标签: {restore_dir.name}。"
                    f" 当前位于第 {self.controller.project.current_frame_index + 1} 帧，"
                    f" 已有 {self.controller.project.sublabel_count()} 个区间标签。"
                )
            else:
                self.controller.load_videos(
                    resolved_top_video_path,
                    resolved_wrist_video_path,
                    resolved_left_wrist_video_path,
                    video_segment_paths=resolved_top_video_paths,
                    wrist_video_segment_paths=resolved_wrist_video_paths,
                    left_wrist_video_segment_paths=resolved_left_wrist_video_paths,
                    source_dataset_root=source_dataset_root,
                    preserve_project=False,
                )
                self.sync_export_settings(
                    dataset_name,
                    instruction_template,
                    effective_input_template,
                )
                # Build loaded views text
                loaded_views = ["top"]
                if resolved_wrist_video_path is not None:
                    loaded_views.append("wrist")
                if resolved_left_wrist_video_path is not None:
                    loaded_views.append("left_wrist")
                loaded_views_text = " + ".join(loaded_views)
                segment_count = len(resolved_top_video_paths)
                status_message = (
                    f"已加载 {loaded_views_text} 视频。"
                    f" 已拼接 {segment_count} 段。"
                    f" 当前导出目录: {Path(resolved_export_dir_text).expanduser().resolve()}"
                )
        except Exception as exc:
            self.status_message = f"加载视频失败: {exc}"
            self._invalidate_frame_cache()
            return self

        alignment_note = self.controller.reader.alignment_note.strip()
        if alignment_note:
            status_message = f"{status_message} {alignment_note}"
        if export_dir_auto_managed:
            status_message = (
                f"{status_message} 已按当前视频自动选择数据集目录: "
                f"{Path(resolved_export_dir_text).name}。"
            )
        if resolved_inputs.resolved_from_lerobot_dataset:
            if resolved_left_wrist_video_path is not None:
                status_message = (
                    f"{status_message} 已从 LeRobot 数据集目录自动解析出 top/wrist/left_wrist 三视角。"
                )
            elif resolved_wrist_video_path is not None:
                status_message = (
                    f"{status_message} 已从 LeRobot 数据集目录自动解析出 top/wrist 双视角。"
                )
            else:
                status_message = f"{status_message} 已从 LeRobot 数据集目录自动解析出 top 视频目录。"
        if resolved_inputs.directory_mode:
            mode_note = f" 当前目录内选中视频: {resolved_inputs.selected_relative_path}。"
            if resolved_inputs.auto_selected_relative_path:
                mode_note = (
                    f" 未显式指定相对视频路径，已自动拼接: "
                    f"{resolved_inputs.selected_relative_path}。"
                )
            status_message = f"{status_message}{mode_note}"

        project = self.controller.project
        if project.video_path is not None:
            self.project_path_text = self._default_project_path().as_posix()
            self.status_message = status_message
            self._ensure_selected_label_group()
            self._ensure_range_label_selection()
            self._refresh_cached_preview()
        return self

    def open_project(
        self,
        project_path_text: str,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
    ) -> "GradioLabelingSession":
        """Open a saved project file from the browser UI."""
        self.sync_export_settings(dataset_name, instruction_template, input_template)
        project_path_str = project_path_text.strip()
        if not project_path_str:
            self.status_message = "请输入项目文件路径。"
            return self

        try:
            project = self.controller.load_project_file(Path(project_path_str))
        except Exception as exc:
            self.status_message = f"打开项目失败: {exc}"
            self._invalidate_frame_cache()
            return self

        self.project_path_text = str(Path(project_path_str).expanduser().resolve())
        self.lerobot_dataset_name = ""
        self.top_input_path_text = "" if project.video_path is None else str(project.video_path)
        self.wrist_input_path_text = (
            "" if project.wrist_video_path is None else str(project.wrist_video_path)
        )
        self.selected_relative_path_text = ""
        self.available_relative_paths = []
        if project.video_path is None:
            self.controller.close_video()
            self._invalidate_frame_cache()
            self.status_message = "项目已加载，但项目里还没有 top 视频路径。"
            return self
        if not project.video_path.exists():
            self.controller.close_video()
            self._invalidate_frame_cache()
            self.status_message = f"项目已加载，但 top 视频不存在: {project.video_path}"
            return self
        if project.wrist_video_path is not None and not project.wrist_video_path.exists():
            self.controller.close_video()
            self._invalidate_frame_cache()
            self.status_message = f"项目已加载，但 wrist 视频不存在: {project.wrist_video_path}"
            return self

        try:
            self.controller.load_videos(
                project.video_path,
                project.wrist_video_path,
                project.left_wrist_video_path,
                video_segment_paths=project.video_segment_paths,
                wrist_video_segment_paths=project.wrist_video_segment_paths,
                left_wrist_video_segment_paths=project.left_wrist_video_segment_paths,
                source_dataset_root=project.source_dataset_root,
                preserve_project=True,
            )
        except Exception as exc:
            self._invalidate_frame_cache()
            self.status_message = f"项目已加载，但视频打开失败: {exc}"
            return self

        self._ensure_selected_label_group()
        self._ensure_range_label_selection()
        self._refresh_cached_preview()
        self.status_message = f"项目已加载: {Path(project_path_str).name}"
        return self

    def save_project(
        self,
        project_path_text: str,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
    ) -> "GradioLabelingSession":
        """Save the current project to a JSON file."""
        self.sync_export_settings(dataset_name, instruction_template, input_template)
        target_text = project_path_text.strip()
        if not target_text:
            target_text = self._default_project_path().as_posix()

        try:
            saved_path = self.controller.save_project_file(Path(target_text))
        except Exception as exc:
            self.status_message = f"保存项目失败: {exc}"
            return self

        self.project_path_text = str(saved_path)
        self.status_message = f"项目已保存: {saved_path}"
        return self

    def export_dataset(
        self,
        export_dir_text: str,
        dataset_name: str,
        instruction_template: str,
        input_template: str,
        progress_callback=None,
    ) -> "GradioLabelingSession":
        """Export the current labels as a LLaMA-Factory dataset."""
        self.sync_export_settings(dataset_name, instruction_template, input_template)
        if self.controller.project.video_path is None:
            self.status_message = "请先加载视频。"
            return self
        target_text, export_dir_auto_managed = self._resolve_export_dir_text(
            export_dir_text,
            self.controller.project.video_path,
            lerobot_dataset_name=self.lerobot_dataset_name,
            selected_relative_path=self.selected_relative_path_text,
        )
        self.export_dir_text = target_text
        self.export_dir_auto_managed = export_dir_auto_managed

        try:
            result = self.controller.export_dataset(Path(target_text), progress_callback=progress_callback)
        except Exception as exc:
            self.status_message = f"导出失败: {exc}"
            return self

        split_text = " / ".join(
            f"{split_name}:{split_count}"
            for split_name, split_count in result.split_counts.items()
        )
        status_parts = [
            f"导出完成，共 {result.exported_samples} 条样本。",
            f"切分: {split_text or '未生成独立切分'}。",
            f"dataset.json: {result.dataset_file}。",
            f"报告: {result.report_file}",
        ]
        if result.training_command:
            status_parts.append(f"训练命令: `{result.training_command}`")
        elif result.training_config_error:
            status_parts.append(f"训练配置生成失败: {result.training_config_error}")
        self.status_message = " ".join(status_parts)
        return self

    def add_label(self, label_text: str) -> "GradioLabelingSession":
        """Append a custom label."""
        try:
            self.controller.add_label(label_text)
        except Exception as exc:
            self.status_message = f"添加标签失败: {exc}"
            return self

        self.status_message = (
            f"已添加标签: {label_text.strip()}。当前标签列表已变为自定义，"
            "如果后续还要复用，可以保存成新的标签组。"
        )
        self.status_message = self._with_autosave_status(self.status_message)
        self._ensure_selected_label_group()
        self._ensure_range_label_selection(preferred=label_text.strip())
        return self

    def apply_label_group(self, group_name: str | None) -> "GradioLabelingSession":
        """Apply one saved label group to the current project."""
        target_group_name = (group_name or "").strip() or self._ensure_selected_label_group()
        if not target_group_name:
            self.status_message = "请先选择一个标签组。"
            return self

        try:
            result = self.controller.apply_label_group(target_group_name)
        except Exception as exc:
            self.status_message = f"应用标签组失败: {exc}"
            return self

        self.selected_label_group_name = self._ensure_selected_label_group(preferred=result.group_name)
        self._ensure_range_label_selection()
        self.status_message = f"已应用标签组“{result.group_name}”，共载入 {result.stored_label_count} 个标签。"
        if result.preserved_annotation_labels:
            preserved_text = "、".join(result.preserved_annotation_labels)
            self.status_message += (
                f" 当前项目里已有标注仍在使用这些标签，所以额外保留了: {preserved_text}。"
            )
        self.status_message = self._with_autosave_status(self.status_message)
        return self

    def save_current_labels_as_group(
        self,
        group_name_text: str,
        *,
        overwrite: bool = False,
    ) -> "GradioLabelingSession":
        """Save the current project labels into the global label-group store."""
        fallback_group_name = self._ensure_selected_label_group()
        target_group_name = group_name_text.strip() or (fallback_group_name or "")
        if not target_group_name:
            self.status_message = "请输入标签组名称。"
            return self

        try:
            result = self.controller.save_current_labels_as_group(
                target_group_name,
                overwrite=overwrite,
            )
        except Exception as exc:
            self.status_message = f"保存标签组失败: {exc}"
            return self

        self.selected_label_group_name = self._ensure_selected_label_group(preferred=result.group_name)
        overwrite_text = "已覆盖保存" if result.overwritten else "已保存"
        self.status_message = (
            f"{overwrite_text}标签组“{result.group_name}”，共 {result.label_count} 个标签。"
            f" 存储位置: {self.controller.label_group_store_path}"
        )
        self.status_message = self._with_autosave_status(self.status_message)
        return self

    def delete_label_group(self, group_name: str | None) -> "GradioLabelingSession":
        """Delete one saved label group from the global store."""
        target_group_name = (group_name or "").strip() or self._ensure_selected_label_group()
        if not target_group_name:
            self.status_message = "请先选择要删除的标签组。"
            return self

        try:
            self.controller.delete_label_group(target_group_name)
        except Exception as exc:
            self.status_message = f"删除标签组失败: {exc}"
            return self

        self.selected_label_group_name = None
        self._ensure_selected_label_group()
        self.status_message = self._with_autosave_status(f"已删除标签组“{target_group_name}”。")
        return self

    def assign_label(self, label: str | None) -> "GradioLabelingSession":
        """Assign the selected label to the current frame and move forward."""
        if not label:
            self.status_message = "请先选择一个标签。"
            return self
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self

        self.controller.assign_label(label)
        self.range_label_selection = self._ensure_range_label_selection(preferred=label)
        if self.controller.project.current_frame_index < self.controller.frame_count - 1:
            self._set_current_frame(self.controller.step_forward_index())
            self.status_message = self._with_autosave_status(
                f"已标注为“{label}”，并按步长 {self.controller.frame_step} 跳到下一帧。"
            )
        else:
            self._refresh_cached_preview()
            self.status_message = self._with_autosave_status(f"最后一帧已标注为“{label}”。")
        return self

    def clear_current_annotation(self) -> "GradioLabelingSession":
        """Remove the current frame label."""
        self.controller.clear_current_annotation()
        self._refresh_cached_preview()
        self.status_message = self._with_autosave_status("已清除当前帧标签。")
        return self

    def go_previous_frame(self) -> "GradioLabelingSession":
        """Move to the previous frame."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        self._set_current_frame(self.controller.step_backward_index())
        self.status_message = f"已按步长 {self.controller.frame_step} 切换到上一帧。"
        return self

    def go_next_frame(self) -> "GradioLabelingSession":
        """Move to the next frame."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        self._set_current_frame(self.controller.step_forward_index())
        self.status_message = f"已按步长 {self.controller.frame_step} 切换到下一帧。"
        return self

    def jump_to_next_unlabeled(self) -> "GradioLabelingSession":
        """Jump to the next unlabeled frame according to the current step size."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self

        next_index = self.controller.find_next_unlabeled_index(use_frame_step=True)
        if next_index is None:
            self.status_message = (
                f"从当前帧往后，按每 {self.controller.frame_step} 帧扫描，没有找到未标注帧。"
            )
            return self

        self._set_current_frame(next_index)
        self.status_message = (
            f"已跳到下一张未标注帧: 第 {self.controller.project.current_frame_index + 1} 帧。"
        )
        return self

    def jump_to_next_keyframe(self) -> "GradioLabelingSession":
        """Jump to the next sufficiently distinct frame."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self

        next_index = self.controller.find_next_distinct_frame(use_frame_step=True)
        if next_index is None:
            self.status_message = (
                f"从当前帧往后，按每 {self.controller.frame_step} 帧扫描，"
                f"没有找到差异达到 {self.controller.keyframe_threshold:.2f} 的关键帧。"
            )
            return self

        self._set_current_frame(next_index)
        self.status_message = (
            f"已跳到下一关键帧: 第 {self.controller.project.current_frame_index + 1} 帧。"
        )
        return self

    def apply_range_label(
        self,
        start_frame_index: int | float,
        end_frame_index: int | float,
        label: str | None,
    ) -> "GradioLabelingSession":
        """Apply one label to a frame interval using the current step size."""
        if not label:
            self.status_message = "请先为区间打标签选择一个标签。"
            return self
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self

        try:
            result = self.controller.apply_label_to_range(
                int(start_frame_index),
                int(end_frame_index),
                label,
                use_frame_step=True,
            )
        except Exception as exc:
            self.status_message = f"区间打标签失败: {exc}"
            return self

        self.range_label_selection = self._ensure_range_label_selection(preferred=label)
        self._set_current_frame(result.last_frame_index)
        self.status_message = self._with_autosave_status(
            f"已按步长 {result.step_used} 将“{label}”应用到第 "
            f"{result.start_index + 1} 到 {result.end_index + 1} 帧，"
            f"覆盖 {result.visited_frames} 帧，更新 {result.updated_frames} 帧。"
        )
        return self

    def jump_to_frame(self, frame_index: int) -> "GradioLabelingSession":
        """Jump to a specific frame index from the slider."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        self._set_current_frame(frame_index)
        self.status_message = f"已跳转到第 {self.controller.project.current_frame_index + 1} 帧。"
        return self

    def jump_to_seconds(self, seconds: float | int) -> "GradioLabelingSession":
        """Jump to the nearest frame on the unified seconds timeline."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        fps = self.controller.reader.fps
        if fps <= 0:
            self.status_message = "无法获取视频帧率。"
            return self
        frame_index = int(round(float(seconds) * fps))
        self._set_current_frame(frame_index)
        current_seconds = self.controller.reader.timestamp_seconds(self.controller.project.current_frame_index)
        self.status_message = f"已跳转到 {current_seconds:.3f} 秒。"
        return self

    def current_seconds(self) -> float:
        """Return the current timeline position in seconds."""
        if self.controller.frame_count <= 0:
            return 0.0
        return self.controller.reader.timestamp_seconds(self.controller.project.current_frame_index)

    def delete_sublabel(self, row_index: int | float) -> "GradioLabelingSession":
        """Delete one saved seconds-based interval label."""
        try:
            removed = self.controller.delete_sublabel(int(row_index))
        except Exception as exc:
            self.status_message = f"删除区间标签失败: {exc}"
            return self
        self.status_message = self._with_autosave_status(
            f"已删除区间标签: {removed.start_seconds:.2f}s 到 {removed.end_seconds:.2f}s，"
            f"标签“{removed.label}”。"
        )
        return self

    def snapshot(self) -> ViewSnapshot:
        """Build a UI snapshot from the current project/controller state."""
        project = self.controller.project
        top_frame_image = self._get_cached_preview("top")
        wrist_frame_image = self._get_cached_preview("wrist")
        left_wrist_frame_image = self._get_cached_preview("left_wrist")
        current_label = project.annotation_for(project.current_frame_index)
        range_label = self._ensure_range_label_selection()
        selected_label_group = self._ensure_selected_label_group()
        labeled_count = project.labeled_frame_count()
        sublabel_count = project.sublabel_count()
        frame_count = self.controller.frame_count
        reader = self.controller.reader
        loaded_views = ["top"]
        if project.wrist_video_path is not None:
            loaded_views.append("wrist")
        if project.left_wrist_video_path is not None:
            loaded_views.append("left_wrist")
        loaded_views_text = " + ".join(loaded_views)
        current_seconds = 0.0
        if frame_count > 0:
            frame_number = project.current_frame_index + 1
            timestamp_seconds = self.controller.reader.timestamp_seconds(project.current_frame_index)
            current_seconds = timestamp_seconds
            segment_index, local_frame_index = reader.segment_for_frame(project.current_frame_index)
            frame_info_lines = [
                f"**当前位置**: {timestamp_seconds:.3f}s",
                f"**总时长**: {frame_count / reader.fps:.3f} 秒" if reader.fps > 0 else "**总时长**: 未知",
                f"**已加载视角**: {loaded_views_text}",
                f"**区间标签数**: {sublabel_count}",
            ]
            # Show original frame counts if any mismatch
            frame_counts = [reader.top_frame_count]
            if project.wrist_video_path is not None:
                frame_counts.append(reader.wrist_frame_count)
            if project.left_wrist_video_path is not None:
                frame_counts.append(reader.left_wrist_frame_count)
            if len(frame_counts) > 1 and not all(c == frame_counts[0] for c in frame_counts):
                frame_count_parts = [f"top={reader.top_frame_count}"]
                if project.wrist_video_path is not None:
                    frame_count_parts.append(f"wrist={reader.wrist_frame_count}")
                if project.left_wrist_video_path is not None:
                    frame_count_parts.append(f"left_wrist={reader.left_wrist_frame_count}")
                frame_info_lines.append(f"**原始帧数**: {' / '.join(frame_count_parts)}")
            if self.selected_relative_path_text:
                frame_info_lines.append(f"**目录内视频**: {self.selected_relative_path_text}")
            frame_info = "  \n".join(frame_info_lines)
        else:
            frame_info = "**当前时间**: 未加载视频"

        resolved_top_video_path_text = ""
        if project.video_path is not None:
            resolved_top_video_path_text = str(project.video_path)
        resolved_wrist_video_path_text = ""
        if project.wrist_video_path is not None:
            resolved_wrist_video_path_text = str(project.wrist_video_path)
        resolved_left_wrist_video_path_text = ""
        if project.left_wrist_video_path is not None:
            resolved_left_wrist_video_path_text = str(project.left_wrist_video_path)
        top_input_path_text = self.top_input_path_text
        wrist_input_path_text = self.wrist_input_path_text

        # Simplified project info
        project_markdown = f"**区间标签数**: {sublabel_count}  \n"
        project_markdown += f"**当前标签组**: {project.label_group_name or '自定义 / 未保存'}  \n"
        if self.export_dir_text:
            project_markdown += f"**导出目录**: {self.export_dir_text}  \n"
        if project.source_dataset_root:
            project_markdown += f"**数据集**: {project.source_dataset_root}  \n"

        label_lines = []
        for label in project.labels:
            label_lines.append(f"- `{label}`: {self.controller.label_count(label)}")
        label_summary = "\n".join(label_lines) if label_lines else "- 暂无标签"
        sublabels_table = self._sublabels_table()

        fps = self.controller.reader.fps
        total_seconds = frame_count / fps if fps > 0 else 0.0
        timeline_image = self.build_timeline_image()
        sync_player_html = self.build_sync_player_html()

        return ViewSnapshot(
            top_frame_image=top_frame_image,
            wrist_frame_image=wrist_frame_image,
            left_wrist_frame_image=left_wrist_frame_image,
            frame_index=project.current_frame_index,
            frame_count=frame_count,
            current_seconds=current_seconds,
            frame_info_markdown=frame_info,
            project_markdown=project_markdown,
            label_summary_markdown=label_summary,
            sublabels_table=sublabels_table,
            labels=list(project.labels),
            label_group_names=self.controller.list_label_groups(),
            selected_label=current_label,
            range_label=range_label,
            selected_label_group=selected_label_group,
            active_label_group_name=project.label_group_name,
            status_message=self.status_message,
            top_input_path=top_input_path_text,
            wrist_input_path=wrist_input_path_text,
            selected_relative_path=self.selected_relative_path_text,
            available_relative_paths=list(self.available_relative_paths),
            resolved_top_video_path=resolved_top_video_path_text,
            resolved_wrist_video_path=resolved_wrist_video_path_text,
            resolved_left_wrist_video_path=resolved_left_wrist_video_path_text,
            dataset_name=project.dataset_name,
            instruction_template=project.instruction_template,
            input_template=project.input_template,
            project_path=self.project_path_text,
            export_dir=self.export_dir_text,
            frame_step=self.controller.frame_step,
            keyframe_threshold=self.controller.keyframe_threshold,
            label_group_store_path=str(self.controller.label_group_store_path),
            fps=fps,
            total_seconds=total_seconds,
            timeline_image=timeline_image,
            sync_player_html=sync_player_html,
        )

    def _default_project_path(self) -> Path:
        """Return a reasonable default project path."""
        if self.export_dir_text.strip():
            export_dir = Path(self.export_dir_text).expanduser().resolve()
            dataset_dir_name = export_dir.name or "video_labeler_project"
            return export_dir / f"{dataset_dir_name}_project.json"
        return Path.cwd() / "video_labeler_project.json"

    def _invalidate_frame_cache(self) -> None:
        """Drop the current browser preview cache."""
        self._frame_image_cache = {}
        self._frame_image_cache_index = -1
        self._frame_image_cache_signature = None

    def _ensure_range_label_selection(self, preferred: str | None = None) -> str | None:
        """Keep the range-label dropdown in sync with the available labels."""
        labels = self.controller.project.labels
        if preferred and preferred in labels:
            self.range_label_selection = preferred
            return self.range_label_selection
        if self.range_label_selection in labels:
            return self.range_label_selection
        self.range_label_selection = labels[0] if labels else None
        return self.range_label_selection

    def _ensure_selected_label_group(self, preferred: str | None = None) -> str | None:
        """Keep the label-group dropdown in sync with the persisted label-group store."""
        label_group_names = self.controller.list_label_groups()
        if preferred and preferred in label_group_names:
            self.selected_label_group_name = preferred
            return self.selected_label_group_name

        project_group_name = self.controller.project.label_group_name
        if project_group_name in label_group_names:
            self.selected_label_group_name = project_group_name
            return self.selected_label_group_name

        if self.selected_label_group_name in label_group_names:
            return self.selected_label_group_name

        self.selected_label_group_name = label_group_names[0] if label_group_names else None
        return self.selected_label_group_name

    def _with_autosave_status(self, base_message: str) -> str:
        """Append one auto-save note to a status message when possible."""
        autosave_note = self._autosave_dataset_if_possible()
        if not autosave_note:
            return base_message
        return f"{base_message} {autosave_note}"

    def _autosave_dataset_if_possible(self) -> str:
        """Persist the dataset workspace after state-changing actions when a video is loaded."""
        project = self.controller.project
        if project.video_path is None:
            return ""

        target_path = Path(self.export_dir_text)

        try:
            result = self.controller.save_dataset_workspace(target_path)
        except Exception as exc:
            return f"自动保存失败: {exc}"

        self.export_dir_text = str(result.output_dir)
        return f"已自动保存到数据集目录 {result.output_dir.name}。"

    def _resolve_export_dir_text(
        self,
        export_dir_text: str | None,
        resolved_top_video_path: Path,
        *,
        lerobot_dataset_name: str = "",
        selected_relative_path: str = "",
    ) -> tuple[str, bool]:
        """Resolve the dataset workspace directory text shown in the UI."""
        normalized_text = (export_dir_text or "").strip()
        default_root_text = str(DEFAULT_EXPORT_DIR)
        current_text = self.export_dir_text.strip()
        should_auto_manage = False

        if not normalized_text or normalized_text == default_root_text:
            should_auto_manage = True
        elif self.export_dir_auto_managed and current_text and normalized_text == current_text:
            should_auto_manage = True

        if should_auto_manage:
            suggested_dir = suggest_export_dir_for_video(
                resolved_top_video_path,
                lerobot_dataset_name=lerobot_dataset_name,
                selected_relative_path=selected_relative_path,
            )
            return str(suggested_dir), True

        return normalized_text, False

    def _set_current_frame(self, frame_index: int) -> None:
        """Move to one frame and refresh the browser preview cache."""
        self.controller.read_frames(frame_index)
        self._refresh_cached_preview()

    def _refresh_cached_preview(self) -> None:
        """Decode the current frame once and cache browser-friendly RGB previews."""
        if self.controller.frame_count <= 0:
            self._invalidate_frame_cache()
            return
        try:
            frames_bgr = self.controller.read_frames(self.controller.project.current_frame_index)
        except Exception as exc:
            self.status_message = f"读取当前帧失败: {exc}"
            self._invalidate_frame_cache()
            return

        preview_cache: dict[str, Any] = {}
        for view_name, frame_bgr in frames_bgr.items():
            preview_bgr = self._downscale_for_preview(frame_bgr)
            preview_cache[view_name] = preview_bgr[:, :, ::-1]

        self._frame_image_cache = preview_cache
        self._frame_image_cache_index = self.controller.project.current_frame_index
        self._frame_image_cache_signature = self._video_signature()

    def _get_cached_preview(self, view_name: str) -> Any | None:
        """Return one cached preview, refreshing it only when needed."""
        if self.controller.frame_count <= 0:
            self._invalidate_frame_cache()
            return None
        if not self.controller.reader.has_view(view_name):
            return None

        if (
            not self._frame_image_cache
            or self._frame_image_cache_index != self.controller.project.current_frame_index
            or self._frame_image_cache_signature != self._video_signature()
        ):
            self._refresh_cached_preview()
        return self._frame_image_cache.get(view_name)

    def _video_signature(self) -> tuple[str | None, str | None, str | None]:
        """Return one immutable signature for the currently loaded video paths."""
        project = self.controller.project
        top_video_path = "|".join(str(path) for path in project.video_segment_paths)
        wrist_video_path = "|".join(str(path) for path in project.wrist_video_segment_paths)
        left_wrist_video_path = "|".join(str(path) for path in project.left_wrist_video_segment_paths)
        if not top_video_path:
            top_video_path = "" if project.video_path is None else str(project.video_path)
        if not wrist_video_path:
            wrist_video_path = "" if project.wrist_video_path is None else str(project.wrist_video_path)
        if not left_wrist_video_path:
            left_wrist_video_path = (
                "" if project.left_wrist_video_path is None else str(project.left_wrist_video_path)
            )
        return top_video_path, wrist_video_path, left_wrist_video_path

    def jump_relative_seconds(self, delta_seconds: float) -> "GradioLabelingSession":
        """Jump forward or backward by a specific number of seconds."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        fps = self.controller.reader.fps
        if fps <= 0:
            self.status_message = "无法获取视频帧率。"
            return self
        current_seconds = self.current_seconds()
        target_seconds = max(0.0, current_seconds + delta_seconds)
        target_frame = int(target_seconds * fps)
        target_frame = max(0, min(target_frame, self.controller.frame_count - 1))
        self._set_current_frame(target_frame)
        actual_seconds = self.current_seconds()
        self.status_message = f"已跳转到 {actual_seconds:.2f}s（{'+' if delta_seconds >= 0 else ''}{delta_seconds:.1f}s）"
        return self

    def jump_relative_frames(self, delta_frames: int) -> "GradioLabelingSession":
        """Jump forward or backward by a specific number of frames."""
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        target_frame = self.controller.project.current_frame_index + int(delta_frames)
        target_frame = max(0, min(target_frame, self.controller.frame_count - 1))
        self._set_current_frame(target_frame)
        actual_seconds = self.current_seconds()
        self.status_message = (
            f"已跳转到 {actual_seconds:.3f}s / frame {target_frame}"
            f"（{'+' if delta_frames >= 0 else ''}{int(delta_frames)} 帧）。"
        )
        return self

    def apply_range_label_seconds(
        self,
        start_sec: float,
        end_sec: float,
        label: str | None,
    ) -> "GradioLabelingSession":
        """Apply a label to a time interval specified in seconds."""
        if not label:
            self.status_message = "请先选择标签。"
            return self
        if self.controller.frame_count <= 0:
            self.status_message = "请先加载视频。"
            return self
        try:
            sublabel = self.controller.add_sublabel_seconds(start_sec, end_sec, label)
        except Exception as exc:
            self.status_message = f"区间打标签失败: {exc}"
            return self
        self.range_label_selection = self._ensure_range_label_selection(preferred=label)
        self._set_current_frame(self.controller.project.current_frame_index)
        self.status_message = self._with_autosave_status(
            f"已添加区间标签“{sublabel.label}”: "
            f"{sublabel.start_seconds:.2f}s 到 {sublabel.end_seconds:.2f}s。"
        )
        return self

    def build_timeline_image(self) -> Any | None:
        """Generate a horizontal timeline image showing labeled intervals."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            return None

        project = self.controller.project
        fps = self.controller.reader.fps
        frame_count = self.controller.frame_count
        if frame_count <= 0 or fps <= 0 or (not project.sublabels and not project.annotations):
            return None

        total_seconds = frame_count / fps
        labels = list(project.labels)
        if not labels:
            return None

        cmap = plt.cm.get_cmap("tab10", max(len(labels), 1))
        label_colors = {lbl: cmap(i) for i, lbl in enumerate(labels)}

        fig, ax = plt.subplots(figsize=(14, 1.8))
        ax.set_xlim(0, total_seconds)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("时间（秒）", fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        intervals: list[tuple[float, float, str]] = []
        if project.sublabels:
            intervals = [
                (item.start_seconds, item.end_seconds, item.label)
                for item in project.sublabels
            ]
        else:
            sorted_frames = sorted(project.annotations.keys())
            cur_label = project.annotations[sorted_frames[0]]
            cur_start = sorted_frames[0]
            prev_frame = sorted_frames[0]
            for frame in sorted_frames[1:]:
                lbl = project.annotations[frame]
                gap = frame - prev_frame
                if lbl != cur_label or gap > self.controller.frame_step * 4:
                    intervals.append((cur_start / fps, prev_frame / fps, cur_label))
                    cur_label = lbl
                    cur_start = frame
                prev_frame = frame
            intervals.append((cur_start / fps, prev_frame / fps, cur_label))

        for start_s, end_s, lbl in intervals:
            color = label_colors.get(lbl, "gray")
            width = max(end_s - start_s, total_seconds * 0.003)
            ax.barh(0, width, left=start_s, height=0.8, color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.3)

        current_sec = self.controller.reader.timestamp_seconds(project.current_frame_index)
        ax.axvline(x=current_sec, color="red", linewidth=1.5, zorder=5)
        ax.text(current_sec, 0.45, f"{current_sec:.1f}s",
                color="red", fontsize=7, ha="center", va="bottom")

        patches = [mpatches.Patch(color=label_colors[lbl], label=lbl)
                   for lbl in labels if lbl in label_colors]
        if patches:
            ax.legend(handles=patches, loc="upper right", fontsize=7,
                      ncol=min(len(patches), 5), framealpha=0.7)

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = rgba[:, :, :3].copy().reshape(h, w, 3)
        plt.close(fig)
        return img

    def _sublabels_table(self) -> list[list[object]]:
        """Return sublabels as rows for a Gradio dataframe."""
        rows: list[list[object]] = []
        fps = self.controller.reader.fps
        frame_count = self.controller.frame_count
        for index, item in enumerate(self.controller.project.sublabels):
            covered_frame_count = len(
                frame_indices_for_seconds_range(
                    item.start_seconds,
                    item.end_seconds,
                    fps=fps,
                    frame_count=frame_count,
                )
            )
            rows.append(
                [
                    index,
                    round(item.start_seconds, 3),
                    round(item.end_seconds, 3),
                    covered_frame_count,
                    item.label,
                ]
            )
        return rows

    def build_sync_player_html(self) -> str:
        """Build the custom synchronized three-view video player."""
        if self.controller.frame_count <= 0:
            return "<div class='vlm-sync-empty'>请先加载数据集。</div>"

        reader = self.controller.reader
        total_seconds = self.controller.frame_count / reader.fps if reader.fps > 0 else 0.0
        current_seconds = self.current_seconds()
        segment_index, local_frame_index = reader.segment_for_frame(self.controller.project.current_frame_index)
        local_seconds = local_frame_index / reader.fps if reader.fps > 0 else 0.0
        segments = reader.segment_metadata()
        segment = segments[segment_index] if 0 <= segment_index < len(segments) else None
        paths = {} if segment is None else segment.get("paths", {})
        view_offset_seconds = (
            {} if segment is None else segment.get("view_offset_seconds", {})
        )
        active_view_list = list(reader.active_views)
        view_offsets_ordered = [
            float(view_offset_seconds.get(view_name, 0.0)) for view_name in active_view_list
        ]
        payload = {
            "segments": segments,
            "views": active_view_list,
            "viewOffsets": view_offsets_ordered,
            "totalSeconds": total_seconds,
            "currentSeconds": current_seconds,
            "localSeconds": local_seconds,
        }
        payload_json = html.escape(json.dumps(payload, ensure_ascii=False), quote=True)
        view_labels = {
            "top": "top",
            "wrist": "right_wrist",
            "left_wrist": "left_wrist",
        }
        video_panels: list[str] = []
        for view_name in reader.active_views:
            path = paths.get(view_name, "") if isinstance(paths, dict) else ""
            source = ""
            if path:
                view_target_seconds = local_seconds + float(
                    view_offset_seconds.get(view_name, 0.0)
                    if isinstance(view_offset_seconds, dict)
                    else 0.0
                )
                source = (
                    f"/gradio_api/file={quote(str(path), safe='')}"
                    f"#t={max(0.0, view_target_seconds):.3f}"
                )
            video_panels.append(
                "<div class=\"vlm-sync-panel\">"
                f"<div class=\"vlm-sync-title\">{html.escape(view_labels.get(view_name, view_name))}</div>"
                f"<video muted playsinline preload=\"auto\" src=\"{html.escape(source, quote=True)}\"></video>"
                "</div>"
            )
        panels_html = "\n".join(video_panels)
        return f"""
<div id="vlm-sync-player" class="vlm-sync-player" data-payload="{payload_json}">
  <div class="vlm-sync-videos">
    {panels_html}
  </div>
  <div class="vlm-sync-controls">
    <button class="vlm-sync-play" id="vlm-play-btn">播放</button>
    <input type="range" class="vlm-sync-range" id="vlm-range" min="0" max="{total_seconds:.3f}" step="0.05" value="{current_seconds:.3f}">
    <div class="vlm-sync-time" id="vlm-time">{current_seconds:.2f}s / {total_seconds:.2f}s</div>
  </div>
</div>
<script>
(() => {{
  const root = document.getElementById("vlm-sync-player");
  if (!root || root.dataset.ready === "1") return;
  root.dataset.ready = "1";
  const payload = JSON.parse(root.dataset.payload || "{{}}");
  const localSeconds = Number(payload.localSeconds || 0);
  const totalSeconds = Number(payload.totalSeconds || 0);
  const viewOffsets = Array.isArray(payload.viewOffsets) ? payload.viewOffsets.map(Number) : [];
  const videos = Array.from(root.querySelectorAll("video"));
  const offsetFor = (i) => Number.isFinite(viewOffsets[i]) ? viewOffsets[i] : 0;
  const playBtn = document.getElementById("vlm-play-btn");
  const rangeInput = document.getElementById("vlm-range");
  const timeDisplay = document.getElementById("vlm-time");
  let isPlaying = false;

  // 确保至少有一个视频
  if (videos.length === 0) {{
    console.error("未找到视频元素");
    return;
  }}

  // 同步播放/暂停
  const syncPlayPause = async (play) => {{
    console.log("syncPlayPause called, play:", play, "videos count:", videos.length);
    isPlaying = play;
    playBtn.textContent = play ? "暂停" : "播放";
    if (play) {{
      for (let i = 0; i < videos.length; i++) {{
        const v = videos[i];
        console.log(`Video ${{i}}: readyState=${{v.readyState}}, paused=${{v.paused}}, src=${{v.src}}`);
        try {{
          const playPromise = v.play();
          console.log(`Video ${{i}} play() called, promise:`, playPromise);
          await playPromise;
          console.log(`Video ${{i}} playing successfully`);
        }} catch (err) {{
          console.error(`Video ${{i}} 播放失败:`, err);
          alert(`视频 ${{i}} 播放失败: ${{err.message}}`);
          isPlaying = false;
          playBtn.textContent = "播放";
          break;
        }}
      }}
    }} else {{
      videos.forEach(v => v.pause());
    }}
  }};

  // 播放按钮
  playBtn?.addEventListener("click", () => syncPlayPause(!isPlaying));

  // 进度条拖动 - 滑块值是 episode 内的逻辑秒，需要加每路视频的物理偏移
  rangeInput?.addEventListener("input", (e) => {{
    const t = Number(e.target.value);
    videos.forEach((v, i) => {{ v.currentTime = offsetFor(i) + t; }});
    if (timeDisplay) timeDisplay.textContent = `${{t.toFixed(2)}}s / ${{totalSeconds.toFixed(2)}}s`;
  }});

  // 同步初始时间 - 每路视频按自己的偏移量定位
  videos.forEach((video, i) => {{
    const target = offsetFor(i) + localSeconds;
    const syncTime = () => {{
      if (Number.isFinite(target) && Math.abs(video.currentTime - target) > 0.08) {{
        video.currentTime = target;
        console.log(`Synced video ${{i}} to ${{target}}s (offset=${{offsetFor(i)}})`);
      }}
    }};

    video.addEventListener("loadedmetadata", syncTime, {{ once: true }});
    video.addEventListener("canplay", syncTime, {{ once: true }});

    if (video.readyState >= 2) {{
      syncTime();
    }}

    // 更新进度条 - 显示 episode 内的逻辑秒（从0开始）
    video.addEventListener("timeupdate", () => {{
      const logicalT = video.currentTime - offsetFor(i);
      if (rangeInput && !rangeInput.matches(":focus")) {{
        rangeInput.value = logicalT;
      }}
      if (timeDisplay) {{
        timeDisplay.textContent = `${{logicalT.toFixed(2)}}s / ${{totalSeconds.toFixed(2)}}s`;
      }}
    }});
  }});

  // 同步所有视频的播放位置 - 以主视频为基准，按各自偏移量定位
  const primary = videos[0];
  if (primary) {{
    primary.addEventListener("play", () => {{
      videos.forEach((v, j) => {{
        if (j > 0) v.play().catch(err => console.error("同步播放失败:", err));
      }});
    }});
    primary.addEventListener("pause", () => {{
      videos.forEach((v, j) => {{ if (j > 0) v.pause(); }});
    }});
    primary.addEventListener("seeked", () => {{
      const primaryLogical = primary.currentTime - offsetFor(0);
      videos.forEach((v, j) => {{
        if (j > 0) v.currentTime = offsetFor(j) + primaryLogical;
      }});
    }});
  }}
}})();
</script>
"""

    @staticmethod
    def _downscale_for_preview(frame_bgr) -> Any:
        """Resize large frames before sending them to the browser."""
        height, width = frame_bgr.shape[:2]
        longest_edge = max(height, width)
        if longest_edge <= MAX_PREVIEW_EDGE:
            return frame_bgr

        scale = MAX_PREVIEW_EDGE / float(longest_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(
            frame_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
