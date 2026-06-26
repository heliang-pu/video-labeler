"""Resolve user-entered video inputs into concrete single-view or paired files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


@dataclass(slots=True)
class ResolvedVideoInputs:
    """Concrete video selection resolved from file or directory mode."""

    top_source_text: str
    wrist_source_text: str
    lerobot_dataset_name: str
    selected_relative_path: str
    available_relative_paths: list[str]
    resolved_top_video_path: Path
    resolved_wrist_video_path: Path | None
    resolved_left_wrist_video_path: Path | None
    resolved_top_video_paths: list[Path]
    resolved_wrist_video_paths: list[Path]
    resolved_left_wrist_video_paths: list[Path]
    source_dataset_root: Path | None
    directory_mode: bool
    auto_selected_relative_path: bool
    resolved_from_lerobot_dataset: bool


def resolve_video_inputs(
    top_path_text: str,
    wrist_path_text: str = "",
    relative_path_text: str = "",
) -> ResolvedVideoInputs:
    """Resolve file-mode or directory-mode UI inputs into concrete video paths."""
    normalized_top_text = top_path_text.strip()
    normalized_wrist_text = wrist_path_text.strip()
    normalized_relative_path = _normalize_relative_path(relative_path_text)

    if not normalized_top_text:
        raise ValueError("请输入 top 视频路径。")

    resolved_top_path = Path(normalized_top_text).expanduser().resolve()
    if not resolved_top_path.exists():
        raise ValueError(f"top 路径不存在: {resolved_top_path}")

    resolved_wrist_path = None
    if normalized_wrist_text:
        resolved_wrist_path = Path(normalized_wrist_text).expanduser().resolve()
        if not resolved_wrist_path.exists():
            raise ValueError(f"wrist 路径不存在: {resolved_wrist_path}")

    lerobot_dataset_name = ""
    resolved_from_lerobot_dataset = False
    resolved_left_wrist_path = None
    original_top_path = resolved_top_path
    lerobot_video_dirs = _resolve_lerobot_video_dirs(resolved_top_path)
    source_dataset_root: Path | None = None
    if lerobot_video_dirs is not None:
        lerobot_top_dir, lerobot_wrist_dir, lerobot_left_wrist_dir = lerobot_video_dirs
        if resolved_wrist_path is not None:
            allowed_wrist_paths = {original_top_path, lerobot_top_dir}
            if lerobot_wrist_dir is not None:
                allowed_wrist_paths.add(lerobot_wrist_dir)
            if resolved_wrist_path not in allowed_wrist_paths:
                raise ValueError(
                    "当 top 输入是 LeRobot 数据集目录时，wrist 请留空，或填写同一个"
                    " LeRobot 数据集目录 / 自动解析出的 wrist 目录。"
                )
        resolved_top_path = lerobot_top_dir
        resolved_wrist_path = lerobot_wrist_dir
        resolved_left_wrist_path = lerobot_left_wrist_dir
        lerobot_dataset_name = original_top_path.name
        resolved_from_lerobot_dataset = True
        source_dataset_root = original_top_path

    if resolved_top_path.is_file():
        _validate_video_file_candidate(resolved_top_path, view_name="top")
        if resolved_wrist_path is not None and resolved_wrist_path.is_dir():
            raise ValueError("top 是文件时，wrist 也必须是文件，不能是目录。")
        if resolved_wrist_path is not None:
            _validate_video_file_candidate(resolved_wrist_path, view_name="wrist")
        if normalized_relative_path:
            raise ValueError("当前输入是单个视频文件，不需要再填写相对视频路径。")
        return ResolvedVideoInputs(
            top_source_text=normalized_top_text,
            wrist_source_text="" if resolved_wrist_path is None else str(resolved_wrist_path),
            lerobot_dataset_name=lerobot_dataset_name,
            selected_relative_path="",
            available_relative_paths=[],
            resolved_top_video_path=resolved_top_path,
            resolved_wrist_video_path=resolved_wrist_path,
            resolved_left_wrist_video_path=resolved_left_wrist_path,
            resolved_top_video_paths=[resolved_top_path],
            resolved_wrist_video_paths=[] if resolved_wrist_path is None else [resolved_wrist_path],
            resolved_left_wrist_video_paths=(
                [] if resolved_left_wrist_path is None else [resolved_left_wrist_path]
            ),
            source_dataset_root=source_dataset_root or _infer_dataset_root(resolved_top_path),
            directory_mode=False,
            auto_selected_relative_path=False,
            resolved_from_lerobot_dataset=resolved_from_lerobot_dataset,
        )

    if not resolved_top_path.is_dir():
        raise ValueError(f"top 输入既不是有效文件也不是有效目录: {resolved_top_path}")

    if resolved_wrist_path is not None and not resolved_wrist_path.is_dir():
        raise ValueError("top 是目录时，wrist 也必须是目录，不能是单个文件。")

    top_relative_paths = _list_relative_video_paths(resolved_top_path)
    if not top_relative_paths:
        raise ValueError(f"top 目录里没有找到可用的非空视频文件: {resolved_top_path}")

    available_relative_paths = list(top_relative_paths)
    if resolved_wrist_path is not None:
        wrist_relative_paths = set(_list_relative_video_paths(resolved_wrist_path))
        available_relative_paths = [
            relative_path
            for relative_path in top_relative_paths
            if relative_path in wrist_relative_paths
        ]
        if not available_relative_paths:
            raise ValueError(
                "top 和 wrist 目录里没有找到同名同相对路径、且都非空可读的视频对。"
            )
    if resolved_left_wrist_path is not None:
        left_wrist_relative_paths = set(_list_relative_video_paths(resolved_left_wrist_path))
        available_relative_paths = [
            relative_path
            for relative_path in available_relative_paths
            if relative_path in left_wrist_relative_paths
        ]
        if not available_relative_paths:
            raise ValueError(
                "top、wrist 和 left_wrist 目录里没有找到同名同相对路径、且都非空可读的视频。"
            )

    auto_selected_relative_path = False
    selected_relative_paths = [normalized_relative_path] if normalized_relative_path else available_relative_paths
    selected_relative_path = normalized_relative_path
    if not normalized_relative_path:
        auto_selected_relative_path = True
        selected_relative_path = f"全部视频（{len(selected_relative_paths)} 段）"

    missing_relative_paths = [
        relative_path
        for relative_path in selected_relative_paths
        if relative_path not in available_relative_paths
    ]
    if missing_relative_paths:
        preview_paths = "，".join(available_relative_paths[:5])
        raise ValueError(
            "输入的相对视频路径不存在于目录配对列表中。"
            f" 目标: {missing_relative_paths[0]}。"
            f" 可用示例: {preview_paths}"
        )

    resolved_top_video_paths: list[Path] = []
    resolved_wrist_video_paths: list[Path] = []
    resolved_left_wrist_video_paths: list[Path] = []
    for relative_path in selected_relative_paths:
        top_video_path = (resolved_top_path / relative_path).resolve()
        _validate_video_file_candidate(top_video_path, view_name="top")
        resolved_top_video_paths.append(top_video_path)
        if resolved_wrist_path is not None:
            wrist_video_path = (resolved_wrist_path / relative_path).resolve()
            _validate_video_file_candidate(wrist_video_path, view_name="wrist")
            resolved_wrist_video_paths.append(wrist_video_path)
        if resolved_left_wrist_path is not None:
            left_wrist_video_path = (resolved_left_wrist_path / relative_path).resolve()
            _validate_video_file_candidate(left_wrist_video_path, view_name="left_wrist")
            resolved_left_wrist_video_paths.append(left_wrist_video_path)

    resolved_top_video_path = resolved_top_video_paths[0]
    resolved_wrist_video_path = resolved_wrist_video_paths[0] if resolved_wrist_video_paths else None
    resolved_left_wrist_video_path = (
        resolved_left_wrist_video_paths[0] if resolved_left_wrist_video_paths else None
    )

    return ResolvedVideoInputs(
        top_source_text=normalized_top_text,
        wrist_source_text=normalized_wrist_text,
        lerobot_dataset_name=lerobot_dataset_name,
        selected_relative_path=selected_relative_path,
        available_relative_paths=available_relative_paths,
        resolved_top_video_path=resolved_top_video_path,
        resolved_wrist_video_path=resolved_wrist_video_path,
        resolved_left_wrist_video_path=resolved_left_wrist_video_path,
        resolved_top_video_paths=resolved_top_video_paths,
        resolved_wrist_video_paths=resolved_wrist_video_paths,
        resolved_left_wrist_video_paths=resolved_left_wrist_video_paths,
        source_dataset_root=source_dataset_root or _infer_dataset_root(resolved_top_path),
        directory_mode=True,
        auto_selected_relative_path=auto_selected_relative_path,
        resolved_from_lerobot_dataset=resolved_from_lerobot_dataset,
    )


def _list_relative_video_paths(root_dir: Path) -> list[str]:
    """Return all supported video files under one directory as POSIX relative paths."""
    relative_paths: list[str] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        if path.stat().st_size <= 0:
            continue
        relative_paths.append(path.relative_to(root_dir).as_posix())
    return relative_paths


def _normalize_relative_path(relative_path_text: str) -> str:
    """Normalize one UI-entered relative path into a stable POSIX path."""
    normalized_text = relative_path_text.strip().replace("\\", "/").strip("/")
    if not normalized_text:
        return ""
    return Path(normalized_text).as_posix()


def _validate_video_file_candidate(video_path: Path, *, view_name: str) -> None:
    """Fail fast on empty or obviously invalid file candidates."""
    if not video_path.exists():
        raise ValueError(f"{view_name} 视频不存在: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"{view_name} 路径不是文件: {video_path}")
    if video_path.stat().st_size <= 0:
        raise ValueError(f"{view_name} 视频文件大小为 0，无法读取: {video_path}")


def _infer_dataset_root(path: Path) -> Path | None:
    """Infer a LeRobot-style dataset root from a camera dir or video file."""
    current = path.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        if parent.name == "videos":
            return parent.parent
    return None


def _resolve_lerobot_video_dirs(dataset_root: Path) -> tuple[Path, Path | None, Path | None] | None:
    """Return top/wrist/left_wrist video directories when one LeRobot dataset root is provided."""
    if not dataset_root.is_dir():
        return None

    # Try standard LeRobot camera naming (camera_top, camera_right_wrist)
    top_dir = dataset_root / "videos" / "observation.images.camera_top"
    if not top_dir.is_dir():
        # Try AGI arm bot naming (top, right_wrist, left_wrist)
        top_dir = dataset_root / "videos" / "observation.images.top"
        if not top_dir.is_dir():
            return None

    # Detect right wrist camera
    wrist_dir = dataset_root / "videos" / "observation.images.camera_right_wrist"
    if not wrist_dir.is_dir():
        wrist_dir = dataset_root / "videos" / "observation.images.right_wrist"
        if not wrist_dir.is_dir():
            # Try generic wrist name
            wrist_dir = dataset_root / "videos" / "observation.images.wrist"
            if not wrist_dir.is_dir():
                wrist_dir = None

    # Detect left wrist camera (for dual-arm robots)
    left_wrist_dir = dataset_root / "videos" / "observation.images.left_wrist"
    if not left_wrist_dir.is_dir():
        left_wrist_dir = None

    return top_dir, wrist_dir, left_wrist_dir
