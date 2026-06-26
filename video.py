"""Video decoding helpers used by the browser labeler and exporter."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    cv2 = None


class VideoDependencyError(RuntimeError):
    """Raised when video decoding dependencies are unavailable."""


class _SingleVideoFrameReader:
    """Thin random-access wrapper around OpenCV's VideoCapture."""

    def __init__(
        self,
        *,
        max_cache_size: int = 8,
        signature_size: int = 32,
        max_signature_cache_size: int = 64,
    ) -> None:
        self._capture: Any | None = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self._last_frame_index = -1
        self._max_cache_size = max(1, int(max_cache_size))
        self._signature_size = max(8, int(signature_size))
        self._max_signature_cache_size = max(1, int(max_signature_cache_size))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._signature_cache: OrderedDict[int, Any] = OrderedDict()

    def open(self, video_path: Path) -> None:
        """Open a video file and read its basic metadata."""
        if cv2 is None:
            raise VideoDependencyError(
                "OpenCV is not installed. Install dependencies from requirements.txt first."
            )

        self.close()
        resolved_path = video_path.expanduser().resolve()
        if not resolved_path.exists():
            raise RuntimeError(f"Failed to open video: {resolved_path}")
        if not resolved_path.is_file():
            raise RuntimeError(f"Video path is not a file: {resolved_path}")
        if resolved_path.stat().st_size <= 0:
            raise RuntimeError(f"Video file is empty: {resolved_path}")
        capture = cv2.VideoCapture(str(resolved_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {resolved_path}")

        self._capture = capture
        self.video_path = resolved_path
        self.frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._last_frame_index = -1
        self._cache.clear()
        self._signature_cache.clear()

    def close(self) -> None:
        """Release any currently opened video file."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self._last_frame_index = -1
        self._cache.clear()
        self._signature_cache.clear()

    def timestamp_seconds(self, frame_index: int) -> float:
        """Translate a frame index into seconds using the container FPS."""
        if self.fps <= 0:
            return 0.0
        return float(frame_index) / self.fps

    def read_frame_bgr(self, frame_index: int) -> Any:
        """Read one frame as a BGR ndarray."""
        if self._capture is None:
            raise RuntimeError("No video is currently loaded.")
        if self.frame_count <= 0:
            raise RuntimeError("The loaded video does not expose a valid frame count.")

        index = max(0, min(int(frame_index), self.frame_count - 1))
        cached_frame = self._cache.get(index)
        if cached_frame is not None:
            self._cache.move_to_end(index)
            return cached_frame.copy()

        if index != self._last_frame_index + 1:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))

        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Failed to decode frame {index} from {self.video_path}")

        self._last_frame_index = index
        self._cache[index] = frame.copy()
        self._cache.move_to_end(index)
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
        return frame

    def read_frame_signature(self, frame_index: int) -> Any:
        """Read one frame signature for fast similarity checks."""
        if cv2 is None:
            raise VideoDependencyError(
                "OpenCV is not installed. Install dependencies from requirements.txt first."
            )
        if self.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")

        index = max(0, min(int(frame_index), self.frame_count - 1))
        cached_signature = self._signature_cache.get(index)
        if cached_signature is not None:
            self._signature_cache.move_to_end(index)
            return cached_signature.copy()

        frame = self.read_frame_bgr(index)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signature = cv2.resize(
            grayscale,
            (self._signature_size, self._signature_size),
            interpolation=cv2.INTER_AREA,
        ).astype("float32")

        self._signature_cache[index] = signature.copy()
        self._signature_cache.move_to_end(index)
        while len(self._signature_cache) > self._max_signature_cache_size:
            self._signature_cache.popitem(last=False)
        return signature

    def frame_difference_score(self, reference_index: int, candidate_index: int) -> float:
        """Return the mean absolute difference between two lightweight frame signatures."""
        if cv2 is None:
            raise VideoDependencyError(
                "OpenCV is not installed. Install dependencies from requirements.txt first."
            )

        reference_signature = self.read_frame_signature(reference_index)
        candidate_signature = self.read_frame_signature(candidate_index)
        difference = cv2.absdiff(reference_signature, candidate_signature)
        return float(difference.mean())


class _SegmentedViewReader:
    """Random-access reader for one view made of sequential video segments."""

    def __init__(self, **reader_kwargs: int) -> None:
        self._reader_kwargs = dict(reader_kwargs)
        self._readers: list[_SingleVideoFrameReader] = []
        self.video_paths: list[Path] = []
        self.frame_counts: list[int] = []
        self.start_frames: list[int] = []
        self.frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0

    def open(self, video_paths: list[Path]) -> None:
        """Open all segments for one camera view."""
        self.close()
        if not video_paths:
            raise RuntimeError("At least one video segment is required.")

        readers: list[_SingleVideoFrameReader] = []
        try:
            for video_path in video_paths:
                reader = _SingleVideoFrameReader(**self._reader_kwargs)
                reader.open(video_path)
                readers.append(reader)
        except Exception:
            for reader in readers:
                reader.close()
            raise

        self._readers = readers
        self.video_paths = [reader.video_path for reader in readers if reader.video_path is not None]
        self.frame_counts = [reader.frame_count for reader in readers]
        self.start_frames = []
        running_frame_count = 0
        for frame_count in self.frame_counts:
            self.start_frames.append(running_frame_count)
            running_frame_count += frame_count

        first_reader = readers[0]
        self.frame_count = running_frame_count
        self.fps = first_reader.fps
        self.frame_width = first_reader.frame_width
        self.frame_height = first_reader.frame_height
        self._validate_segments()

    def close(self) -> None:
        """Release every segment decoder."""
        for reader in self._readers:
            reader.close()
        self._readers = []
        self.video_paths = []
        self.frame_counts = []
        self.start_frames = []
        self.frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0

    def timestamp_seconds(self, frame_index: int) -> float:
        """Translate one global frame index into seconds."""
        if self.fps <= 0:
            return 0.0
        return float(frame_index) / self.fps

    def segment_for_frame(self, frame_index: int) -> tuple[int, int]:
        """Return ``(segment_index, local_frame_index)`` for a global frame."""
        if self.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")
        bounded_index = max(0, min(int(frame_index), self.frame_count - 1))
        segment_index = bisect_right(self.start_frames, bounded_index) - 1
        segment_index = max(0, min(segment_index, len(self._readers) - 1))
        local_frame_index = bounded_index - self.start_frames[segment_index]
        return segment_index, local_frame_index

    def read_frame_bgr(self, frame_index: int) -> Any:
        """Read one global frame as BGR."""
        segment_index, local_frame_index = self.segment_for_frame(frame_index)
        return self.read_segment_frame_bgr(segment_index, local_frame_index)

    def read_segment_frame_bgr(self, segment_index: int, local_frame_index: int) -> Any:
        """Read one frame by explicit segment and local frame index."""
        if not self._readers:
            raise RuntimeError("No video is currently loaded.")
        segment_index = max(0, min(int(segment_index), len(self._readers) - 1))
        reader = self._readers[segment_index]
        return reader.read_frame_bgr(local_frame_index)

    def read_segment_frame_signature(self, segment_index: int, local_frame_index: int) -> Any:
        """Read one frame signature by explicit segment and local frame index."""
        if not self._readers:
            raise RuntimeError("No video is currently loaded.")
        segment_index = max(0, min(int(segment_index), len(self._readers) - 1))
        reader = self._readers[segment_index]
        return reader.read_frame_signature(local_frame_index)

    def read_frame_signature(self, frame_index: int) -> Any:
        """Read one global-frame signature for fast similarity checks."""
        segment_index, local_frame_index = self.segment_for_frame(frame_index)
        return self._readers[segment_index].read_frame_signature(local_frame_index)

    def frame_difference_score(self, reference_index: int, candidate_index: int) -> float:
        """Return a lightweight frame-difference score across global indices."""
        if cv2 is None:
            raise VideoDependencyError(
                "OpenCV is not installed. Install dependencies from requirements.txt first."
            )

        reference_signature = self.read_frame_signature(reference_index)
        candidate_signature = self.read_frame_signature(candidate_index)
        difference = cv2.absdiff(reference_signature, candidate_signature)
        return float(difference.mean())

    def _validate_segments(self) -> None:
        """Ensure all segments in one view are usable as a logical timeline."""
        if not self._readers:
            return
        reference_fps = self._readers[0].fps
        for reader in self._readers:
            if reader.frame_count <= 0:
                raise ValueError(f"视频无法读取到有效帧数: {reader.video_path}")
            if reader.fps > 0 and reference_fps > 0 and abs(reader.fps - reference_fps) > 1e-3:
                raise ValueError(
                    "同一视角内视频片段 fps 不一致，无法拼接为统一时间轴。 "
                    f"first={reference_fps}, current={reader.fps}, path={reader.video_path}"
                )


class VideoFrameReader:
    """Read one, two, or three synchronized video views with shared frame navigation."""

    def __init__(
        self,
        *,
        max_cache_size: int = 8,
        signature_size: int = 32,
        max_signature_cache_size: int = 64,
    ) -> None:
        self._reader_kwargs = {
            "max_cache_size": max_cache_size,
            "signature_size": signature_size,
            "max_signature_cache_size": max_signature_cache_size,
        }
        self._top_reader = _SegmentedViewReader(**self._reader_kwargs)
        self._wrist_reader = _SegmentedViewReader(**self._reader_kwargs)
        self._left_wrist_reader = _SegmentedViewReader(**self._reader_kwargs)
        self.video_path: Path | None = None
        self.wrist_video_path: Path | None = None
        self.left_wrist_video_path: Path | None = None
        self.video_segment_paths: list[Path] = []
        self.wrist_video_segment_paths: list[Path] = []
        self.left_wrist_video_segment_paths: list[Path] = []
        self.frame_count = 0
        self.top_frame_count = 0
        self.wrist_frame_count = 0
        self.left_wrist_frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.alignment_note = ""
        self._timeline_start_frames: list[int] = []
        self._timeline_segment_frame_counts: list[int] = []
        self.view_frame_offsets: dict[str, int] = {"top": 0, "wrist": 0, "left_wrist": 0}
        self._episode_active = False
        self._episode_frame_count = 0

    @property
    def active_views(self) -> tuple[str, ...]:
        """Return active video view names in deterministic order."""
        views = ["top"]
        if self.wrist_video_path is not None:
            views.append("wrist")
        if self.left_wrist_video_path is not None:
            views.append("left_wrist")
        return tuple(views)

    def open(
        self,
        video_path: Path,
        wrist_video_path: Path | None = None,
        left_wrist_video_path: Path | None = None,
        *,
        video_segment_paths: list[Path] | None = None,
        wrist_video_segment_paths: list[Path] | None = None,
        left_wrist_video_segment_paths: list[Path] | None = None,
    ) -> None:
        """Open one, two, or three synchronized video files and validate metadata."""
        resolved_top_segments = self._resolve_segments(video_path, video_segment_paths)
        resolved_wrist_segments = self._resolve_optional_segments(
            wrist_video_path,
            wrist_video_segment_paths,
        )
        resolved_left_wrist_segments = self._resolve_optional_segments(
            left_wrist_video_path,
            left_wrist_video_segment_paths,
        )
        resolved_top_path = resolved_top_segments[0]
        resolved_wrist_path = resolved_wrist_segments[0] if resolved_wrist_segments else None
        resolved_left_wrist_path = resolved_left_wrist_segments[0] if resolved_left_wrist_segments else None

        self.close()
        self._top_reader.open(resolved_top_segments)
        self.video_path = resolved_top_path
        self.wrist_video_path = resolved_wrist_path
        self.left_wrist_video_path = resolved_left_wrist_path
        self.video_segment_paths = list(resolved_top_segments)
        self.wrist_video_segment_paths = list(resolved_wrist_segments)
        self.left_wrist_video_segment_paths = list(resolved_left_wrist_segments)
        try:
            if resolved_wrist_segments:
                self._wrist_reader.open(resolved_wrist_segments)
            if resolved_left_wrist_segments:
                self._left_wrist_reader.open(resolved_left_wrist_segments)
            if resolved_wrist_segments or resolved_left_wrist_segments:
                self._validate_alignment()
        except Exception:
            self.close()
            raise

        self.top_frame_count = self._top_reader.frame_count
        self.wrist_frame_count = self._wrist_reader.frame_count if resolved_wrist_segments else 0
        self.left_wrist_frame_count = (
            self._left_wrist_reader.frame_count if resolved_left_wrist_segments else 0
        )
        self._rebuild_timeline()
        self.frame_count = sum(self._timeline_segment_frame_counts) or self._top_reader.frame_count
        self.fps = self._top_reader.fps
        self.frame_width = self._top_reader.frame_width
        self.frame_height = self._top_reader.frame_height
        self.alignment_note = ""

        frame_count_info = [f"top={self._top_reader.frame_count}"]
        if resolved_wrist_segments:
            frame_count_info.append(f"wrist={self._wrist_reader.frame_count}")
        if resolved_left_wrist_segments:
            frame_count_info.append(f"left_wrist={self._left_wrist_reader.frame_count}")

        active_total_counts = [
            self._top_reader.frame_count,
            *([self._wrist_reader.frame_count] if resolved_wrist_segments else []),
            *([self._left_wrist_reader.frame_count] if resolved_left_wrist_segments else []),
        ]
        if len(active_total_counts) > 1 and not all(count == active_total_counts[0] for count in active_total_counts):
            self.alignment_note = (
                "检测到视频总帧数不一致，"
                f"已按各同名片段的共同帧数拼接为 {self.frame_count} 帧同步时间轴。 "
                + ", ".join(frame_count_info)
            )
        if len(resolved_top_segments) > 1:
            segment_count_info = [f"top={len(resolved_top_segments)}"]
            if resolved_wrist_segments:
                segment_count_info.append(f"wrist={len(resolved_wrist_segments)}")
            if resolved_left_wrist_segments:
                segment_count_info.append(f"left_wrist={len(resolved_left_wrist_segments)}")
            segment_note = "已将多个视频片段按文件顺序拼成一条同步时间轴。 " + ", ".join(segment_count_info)
            if self.alignment_note:
                self.alignment_note = f"{self.alignment_note} {segment_note}"
            else:
                self.alignment_note = segment_note

    def close(self) -> None:
        """Release any currently opened video files."""
        self._top_reader.close()
        self._wrist_reader.close()
        self._left_wrist_reader.close()
        self.video_path = None
        self.wrist_video_path = None
        self.left_wrist_video_path = None
        self.video_segment_paths = []
        self.wrist_video_segment_paths = []
        self.left_wrist_video_segment_paths = []
        self.frame_count = 0
        self.top_frame_count = 0
        self.wrist_frame_count = 0
        self.left_wrist_frame_count = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.alignment_note = ""
        self._timeline_start_frames = []
        self._timeline_segment_frame_counts = []
        self.view_frame_offsets = {"top": 0, "wrist": 0, "left_wrist": 0}
        self._episode_active = False
        self._episode_frame_count = 0

    def set_episode_window(
        self,
        view_frame_offsets: dict[str, int],
        episode_frame_count: int,
    ) -> None:
        """Constrain the logical timeline to a per-view windowed episode.

        Each view's physical frame index becomes ``logical + offset``.
        Frame counts and segment metadata are reported using the logical
        ``episode_frame_count`` so the UI shows just this episode.
        """
        if self.fps <= 0 or self._top_reader.frame_count <= 0:
            return
        clean_offsets = {
            "top": int(view_frame_offsets.get("top", 0)),
            "wrist": int(view_frame_offsets.get("wrist", 0)),
            "left_wrist": int(view_frame_offsets.get("left_wrist", 0)),
        }
        # Clamp offsets to legal physical ranges and clamp episode_frame_count too.
        bounded_episode_frames = max(1, int(episode_frame_count))
        for view_name, reader in (
            ("top", self._top_reader),
            ("wrist", self._wrist_reader),
            ("left_wrist", self._left_wrist_reader),
        ):
            if reader.frame_count <= 0:
                continue
            clean_offsets[view_name] = max(0, min(clean_offsets[view_name], reader.frame_count - 1))
            available = reader.frame_count - clean_offsets[view_name]
            bounded_episode_frames = min(bounded_episode_frames, max(1, available))

        self.view_frame_offsets = clean_offsets
        self._episode_active = True
        self._episode_frame_count = bounded_episode_frames
        self.frame_count = bounded_episode_frames
        self.top_frame_count = bounded_episode_frames
        if self.wrist_video_path is not None:
            self.wrist_frame_count = bounded_episode_frames
        if self.left_wrist_video_path is not None:
            self.left_wrist_frame_count = bounded_episode_frames
        # Re-anchor the logical timeline to a single virtual segment.
        self._timeline_start_frames = [0]
        self._timeline_segment_frame_counts = [bounded_episode_frames]

    def clear_episode_window(self) -> None:
        """Drop the episode windowing if it was previously set."""
        self.view_frame_offsets = {"top": 0, "wrist": 0, "left_wrist": 0}
        self._episode_active = False
        self._episode_frame_count = 0

    def matches_paths(
        self,
        video_path: Path,
        wrist_video_path: Path | None = None,
        left_wrist_video_path: Path | None = None,
        *,
        video_segment_paths: list[Path] | None = None,
        wrist_video_segment_paths: list[Path] | None = None,
        left_wrist_video_segment_paths: list[Path] | None = None,
    ) -> bool:
        """Return whether the reader is already opened on the given path set."""
        resolved_top_segments = self._resolve_segments(video_path, video_segment_paths)
        resolved_wrist_segments = self._resolve_optional_segments(
            wrist_video_path,
            wrist_video_segment_paths,
        )
        resolved_left_wrist_segments = self._resolve_optional_segments(
            left_wrist_video_path,
            left_wrist_video_segment_paths,
        )
        return (
            self.video_segment_paths == resolved_top_segments
            and self.wrist_video_segment_paths == resolved_wrist_segments
            and self.left_wrist_video_segment_paths == resolved_left_wrist_segments
            and self.frame_count > 0
        )

    def has_view(self, view_name: str) -> bool:
        """Return whether one named view is currently active."""
        if view_name == "top":
            return self.video_path is not None
        if view_name == "wrist":
            return self.wrist_video_path is not None
        if view_name == "left_wrist":
            return self.left_wrist_video_path is not None
        return False

    def active_video_items(self) -> list[tuple[str, Path]]:
        """Return active video paths in deterministic order."""
        items: list[tuple[str, Path]] = []
        if self.video_path is not None:
            items.append(("top", self.video_path))
        if self.wrist_video_path is not None:
            items.append(("wrist", self.wrist_video_path))
        if self.left_wrist_video_path is not None:
            items.append(("left_wrist", self.left_wrist_video_path))
        return items

    def active_video_segments(self) -> dict[str, list[Path]]:
        """Return active segment paths for each view."""
        items: dict[str, list[Path]] = {}
        if self.video_segment_paths:
            items["top"] = list(self.video_segment_paths)
        if self.wrist_video_segment_paths:
            items["wrist"] = list(self.wrist_video_segment_paths)
        if self.left_wrist_video_segment_paths:
            items["left_wrist"] = list(self.left_wrist_video_segment_paths)
        return items

    def segment_metadata(self) -> list[dict[str, object]]:
        """Return logical-timeline segment metadata with per-view paths and offsets."""
        segments: list[dict[str, object]] = []
        if not self.video_segment_paths:
            return segments
        active_segments = self.active_video_segments()
        segment_count = len(self._timeline_start_frames)
        for segment_index in range(segment_count):
            frame_count = self._timeline_segment_frame_counts[segment_index]
            start_frame = self._timeline_start_frames[segment_index]
            start_seconds = self.timestamp_seconds(start_frame)
            duration_seconds = frame_count / self.fps if self.fps > 0 else 0.0
            paths: dict[str, str] = {}
            for view_name, view_paths in active_segments.items():
                if segment_index < len(view_paths):
                    paths[view_name] = str(view_paths[segment_index])
            view_offset_seconds: dict[str, float] = {}
            if self._episode_active and self.fps > 0:
                for view_name, offset_frames in self.view_frame_offsets.items():
                    view_offset_seconds[view_name] = float(offset_frames) / self.fps
            segments.append(
                {
                    "index": segment_index,
                    "start_frame": int(start_frame),
                    "frame_count": int(frame_count),
                    "start_seconds": float(start_seconds),
                    "duration_seconds": float(duration_seconds),
                    "paths": paths,
                    "view_offset_seconds": view_offset_seconds,
                }
            )
        return segments

    def segment_for_frame(self, frame_index: int) -> tuple[int, int]:
        """Return top-view segment and local frame for one global frame."""
        if self.frame_count <= 0:
            raise RuntimeError("No video is currently loaded.")
        bounded_index = max(0, min(int(frame_index), self.frame_count - 1))
        segment_index = bisect_right(self._timeline_start_frames, bounded_index) - 1
        segment_index = max(0, min(segment_index, len(self._timeline_start_frames) - 1))
        local_frame_index = bounded_index - self._timeline_start_frames[segment_index]
        return segment_index, local_frame_index

    def timestamp_seconds(self, frame_index: int) -> float:
        """Translate a frame index into seconds using the top-view FPS."""
        return self._top_reader.timestamp_seconds(frame_index)

    def read_frame_bgr(self, frame_index: int, view_name: str = "top") -> Any:
        """Read one frame from one specific view as a BGR ndarray."""
        reader = self._reader_for_view(view_name)
        segment_index, local_frame_index = self.segment_for_frame(frame_index)
        physical_local = local_frame_index + self.view_frame_offsets.get(view_name, 0)
        return reader.read_segment_frame_bgr(segment_index, physical_local)

    def read_frames_bgr(self, frame_index: int) -> dict[str, Any]:
        """Read one synchronized frame from all active views."""
        segment_index, local_frame_index = self.segment_for_frame(frame_index)
        top_local = local_frame_index + self.view_frame_offsets.get("top", 0)
        frames = {"top": self._top_reader.read_segment_frame_bgr(segment_index, top_local)}
        if self.wrist_video_segment_paths:
            wrist_local = local_frame_index + self.view_frame_offsets.get("wrist", 0)
            frames["wrist"] = self._wrist_reader.read_segment_frame_bgr(segment_index, wrist_local)
        if self.left_wrist_video_segment_paths:
            left_local = local_frame_index + self.view_frame_offsets.get("left_wrist", 0)
            frames["left_wrist"] = self._left_wrist_reader.read_segment_frame_bgr(segment_index, left_local)
        return frames

    def frame_difference_score(self, reference_index: int, candidate_index: int) -> float:
        """Return one combined frame-difference score across active views."""
        scores = [self._frame_difference_score_for_view(self._top_reader, reference_index, candidate_index)]
        if self.wrist_video_segment_paths:
            scores.append(self._frame_difference_score_for_view(self._wrist_reader, reference_index, candidate_index))
        if self.left_wrist_video_segment_paths:
            scores.append(
                self._frame_difference_score_for_view(
                    self._left_wrist_reader,
                    reference_index,
                    candidate_index,
                )
            )
        return float(max(scores))

    def _validate_alignment(self) -> None:
        """Ensure all active videos are safe to navigate with one shared frame index."""
        # Collect all active readers and their FPS
        active_readers = [self._top_reader]
        fps_info = [f"top={self._top_reader.fps}"]
        frame_count_info = [f"top={self._top_reader.frame_count}"]

        if self.wrist_video_segment_paths:
            active_readers.append(self._wrist_reader)
            fps_info.append(f"wrist={self._wrist_reader.fps}")
            frame_count_info.append(f"wrist={self._wrist_reader.frame_count}")

        if self.left_wrist_video_segment_paths:
            active_readers.append(self._left_wrist_reader)
            fps_info.append(f"left_wrist={self._left_wrist_reader.fps}")
            frame_count_info.append(f"left_wrist={self._left_wrist_reader.frame_count}")

        # Validate FPS consistency
        if len(active_readers) > 1:
            reference_fps = active_readers[0].fps
            for reader in active_readers[1:]:
                if reader.fps > 0 and reference_fps > 0:
                    fps_gap = abs(reference_fps - reader.fps)
                    if fps_gap > 1e-3:
                        raise ValueError(
                            f"视频 fps 不一致，无法按逐帧方式同步加载。 {', '.join(fps_info)}"
                        )

        # Validate frame counts
        for reader in active_readers:
            if reader.frame_count <= 0:
                raise ValueError(f"视频无法读取到有效帧数，无法同步加载。 {', '.join(frame_count_info)}")

    def _reader_for_view(self, view_name: str) -> _SegmentedViewReader:
        """Resolve one view name into its concrete reader."""
        if view_name == "top":
            return self._top_reader
        if view_name == "wrist" and self.wrist_video_path is not None:
            return self._wrist_reader
        if view_name == "left_wrist" and self.left_wrist_video_path is not None:
            return self._left_wrist_reader
        raise ValueError(f"Unknown or inactive video view: {view_name}")

    @staticmethod
    def _resolve_segments(
        video_path: Path,
        video_segment_paths: list[Path] | None,
    ) -> list[Path]:
        """Resolve a required top-view segment list."""
        raw_paths = video_segment_paths if video_segment_paths else [video_path]
        return [path.expanduser().resolve() for path in raw_paths]

    @staticmethod
    def _resolve_optional_segments(
        video_path: Path | None,
        video_segment_paths: list[Path] | None,
    ) -> list[Path]:
        """Resolve an optional view segment list."""
        if video_segment_paths:
            return [path.expanduser().resolve() for path in video_segment_paths]
        if video_path is None:
            return []
        return [video_path.expanduser().resolve()]

    def _rebuild_timeline(self) -> None:
        """Build a logical timeline from per-segment shared frame counts."""
        self._timeline_start_frames = []
        self._timeline_segment_frame_counts = []
        running_frame_count = 0
        segment_count = len(self.video_segment_paths)
        for segment_index in range(segment_count):
            counts = [self._top_reader.frame_counts[segment_index]]
            if self.wrist_video_segment_paths:
                counts.append(self._wrist_reader.frame_counts[segment_index])
            if self.left_wrist_video_segment_paths:
                counts.append(self._left_wrist_reader.frame_counts[segment_index])
            shared_count = min(counts)
            self._timeline_start_frames.append(running_frame_count)
            self._timeline_segment_frame_counts.append(shared_count)
            running_frame_count += shared_count

    def _frame_difference_score_for_view(
        self,
        reader: _SegmentedViewReader,
        reference_index: int,
        candidate_index: int,
    ) -> float:
        """Return a difference score using the shared logical timeline."""
        if cv2 is None:
            raise VideoDependencyError(
                "OpenCV is not installed. Install dependencies from requirements.txt first."
            )
        view_name = self._view_name_for_reader(reader)
        view_offset = self.view_frame_offsets.get(view_name, 0)
        ref_segment, ref_local = self.segment_for_frame(reference_index)
        cand_segment, cand_local = self.segment_for_frame(candidate_index)
        reference_signature = reader.read_segment_frame_signature(ref_segment, ref_local + view_offset)
        candidate_signature = reader.read_segment_frame_signature(cand_segment, cand_local + view_offset)
        difference = cv2.absdiff(reference_signature, candidate_signature)
        return float(difference.mean())

    def _view_name_for_reader(self, reader: _SegmentedViewReader) -> str:
        """Resolve the canonical view name for a segmented reader instance."""
        if reader is self._top_reader:
            return "top"
        if reader is self._wrist_reader:
            return "wrist"
        if reader is self._left_wrist_reader:
            return "left_wrist"
        return "top"
