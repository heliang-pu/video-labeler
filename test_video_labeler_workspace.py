"""Script-style smoke tests for dataset-workspace persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from video_labeler.gradio_session import GradioLabelingSession
from video_labeler.models import (
    DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
    DEFAULT_LABEL_GROUP_NAME,
    DEFAULT_LABELS,
    ProjectState,
)
from video_labeler.storage import load_project
from video_labeler.workspace_store import (
    load_project_from_dataset_dir,
    persist_project_workspace,
    workspace_has_state,
)


def _write_demo_video(video_path: Path, *, frame_count: int = 12) -> None:
    """Create a small synthetic MP4 file for smoke tests."""
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create test video: {video_path}")

    for frame_index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (frame_index * 17) % 255
        frame[:, :, 1] = (frame_index * 29) % 255
        frame[:, :, 2] = (frame_index * 43) % 255
        writer.write(frame)
    writer.release()


def main() -> None:
    """Run persistence-focused smoke tests."""
    assert DEFAULT_LABEL_GROUP_NAME == "parcel_sorting_bimanual"
    assert DEFAULT_LABELS == [
        "left_hand_pickup",
        "object_at_center",
        "right_hand_transfer",
    ]
    default_session = GradioLabelingSession()
    default_snapshot = default_session.snapshot()
    assert default_snapshot.selected_label_group == "parcel_sorting_bimanual"
    assert default_snapshot.labels == DEFAULT_LABELS

    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        video_path = temp_dir / "demo_top.mp4"
        wrist_video_path = temp_dir / "demo_wrist.mp4"
        mismatch_top_video_path = temp_dir / "mismatch_top.mp4"
        mismatch_wrist_video_path = temp_dir / "mismatch_wrist.mp4"
        other_video_path = temp_dir / "other_top.mp4"
        other_wrist_video_path = temp_dir / "other_wrist.mp4"
        top_dir = temp_dir / "observation.images.camera_top"
        wrist_dir = temp_dir / "observation.images.camera_right_wrist"
        top_chunk_dir = top_dir / "chunk-000"
        wrist_chunk_dir = wrist_dir / "chunk-000"
        lerobot_root = temp_dir / "demo_lerobot_dataset"
        lerobot_top_dir = lerobot_root / "videos" / "observation.images.camera_top"
        lerobot_wrist_dir = lerobot_root / "videos" / "observation.images.camera_right_wrist"
        lerobot_top_chunk_dir = lerobot_top_dir / "chunk-000"
        lerobot_wrist_chunk_dir = lerobot_wrist_dir / "chunk-000"
        dataset_dir = temp_dir / "demo_workspace"
        _write_demo_video(video_path)
        _write_demo_video(wrist_video_path)
        _write_demo_video(mismatch_top_video_path, frame_count=12)
        _write_demo_video(mismatch_wrist_video_path, frame_count=15)
        _write_demo_video(other_video_path)
        _write_demo_video(other_wrist_video_path)
        top_chunk_dir.mkdir(parents=True, exist_ok=True)
        wrist_chunk_dir.mkdir(parents=True, exist_ok=True)
        lerobot_top_chunk_dir.mkdir(parents=True, exist_ok=True)
        lerobot_wrist_chunk_dir.mkdir(parents=True, exist_ok=True)
        _write_demo_video(top_chunk_dir / "file-000.mp4", frame_count=10)
        _write_demo_video(wrist_chunk_dir / "file-000.mp4", frame_count=10)
        _write_demo_video(top_chunk_dir / "file-001.mp4", frame_count=8)
        _write_demo_video(wrist_chunk_dir / "file-001.mp4", frame_count=8)
        _write_demo_video(lerobot_top_chunk_dir / "file-000.mp4", frame_count=9)
        _write_demo_video(lerobot_wrist_chunk_dir / "file-000.mp4", frame_count=9)
        (top_chunk_dir / "file-999.mp4").write_bytes(b"")
        (wrist_chunk_dir / "file-999.mp4").write_bytes(b"")

        project = ProjectState(
            video_path=video_path,
            wrist_video_path=wrist_video_path,
            labels=["待机", "执行中", "异常"],
            annotations={0: "待机", 3: "执行中", 6: "异常", 9: "执行中", 11: "待机"},
            current_frame_index=10,
            frame_step=3,
            keyframe_threshold=8.5,
            label_group_name=None,
            input_template=DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
            dataset_name="demo labels",
        )
        first_result = persist_project_workspace(project, dataset_dir, require_annotations=True)
        assert first_result.dataset_file.exists()
        assert first_result.annotations_file.exists()
        assert (dataset_dir / "workspace_meta.json").exists()
        assert (dataset_dir / "train.json").exists()
        assert (dataset_dir / "val.json").exists()
        assert (dataset_dir / "test.json").exists()
        assert (dataset_dir / "train_qwen25vl_lora.yaml").exists()
        assert (dataset_dir / "training_readme.md").exists()
        assert first_result.training_config_file == dataset_dir / "train_qwen25vl_lora.yaml"
        assert first_result.training_readme_file == dataset_dir / "training_readme.md"
        assert first_result.training_command == (
            "bash /workspace/video_finetune/run_qwen25vl_lora_train.sh "
            f"{dataset_dir / 'train_qwen25vl_lora.yaml'}"
        )
        assert first_result.training_config_error is None
        training_yaml_text = (dataset_dir / "train_qwen25vl_lora.yaml").read_text(
            encoding="utf-8"
        )
        assert f"dataset_dir: {dataset_dir}" in training_yaml_text
        assert "dataset: demo_labels_train" in training_yaml_text
        assert "eval_dataset: demo_labels_val" in training_yaml_text
        assert "val_size: 0.0" in training_yaml_text
        assert (
            "output_dir: /workspace/video_finetune/outputs/demo_workspace_qwen25vl_lora"
            in training_yaml_text
        )

        dataset_records = json.loads(first_result.dataset_file.read_text(encoding="utf-8"))
        assert len(dataset_records) == 5
        assert dataset_records[0]["input"] == DEFAULT_DUAL_VIEW_INPUT_TEMPLATE
        assert len(dataset_records[0]["images"]) == 2
        assert dataset_records[0]["images"][0].startswith("images/top/")
        assert dataset_records[0]["images"][1].startswith("images/wrist/")

        restored = load_project_from_dataset_dir(
            dataset_dir,
            video_path=video_path,
            wrist_video_path=wrist_video_path,
        )
        assert restored.project.annotations == project.annotations
        assert restored.project.current_frame_index == 10
        assert restored.project.labels == project.labels
        assert restored.project.label_group_name is None
        assert restored.project.dataset_name == "demo labels"
        assert restored.project.wrist_video_path == wrist_video_path

        invalid_project = ProjectState(
            video_path=video_path,
            wrist_video_path=wrist_video_path,
            labels=["待机"],
            annotations={999: "待机"},
        )
        try:
            persist_project_workspace(invalid_project, temp_dir / "invalid_workspace", require_annotations=True)
        except ValueError as exc:
            assert "超出当前视频范围" in str(exc)
        else:
            raise AssertionError("Expected out-of-range annotations to fail.")

        try:
            load_project_from_dataset_dir(
                dataset_dir,
                video_path=other_video_path,
                wrist_video_path=other_wrist_video_path,
            )
        except ValueError as exc:
            assert "输入视频不一致" in str(exc)
        else:
            raise AssertionError("Expected mismatched video restore to fail.")

        try:
            load_project_from_dataset_dir(dataset_dir, video_path=video_path)
        except ValueError as exc:
            assert "wrist 视频" in str(exc)
        else:
            raise AssertionError("Expected missing wrist restore path to fail.")

        project.annotations = {0: "待机", 3: "执行中"}
        project.current_frame_index = 4
        second_result = persist_project_workspace(project, dataset_dir, require_annotations=True)
        assert second_result.exported_samples == 2
        second_yaml_text = (dataset_dir / "train_qwen25vl_lora.yaml").read_text(
            encoding="utf-8"
        )
        assert "dataset: demo_labels_train" in second_yaml_text
        assert "eval_dataset:" not in second_yaml_text
        assert "val_size: 0.0" in second_yaml_text
        assert "eval_strategy: steps" not in second_yaml_text
        assert "per_device_eval_batch_size" not in second_yaml_text
        assert not (dataset_dir / "val.json").exists()
        assert not (dataset_dir / "test.json").exists()
        image_paths = sorted(
            path.relative_to(dataset_dir).as_posix()
            for path in (dataset_dir / "images").rglob("*.jpg")
        )
        assert image_paths == [
            "images/top/demo_top_000000.jpg",
            "images/top/demo_top_000003.jpg",
            "images/wrist/demo_wrist_000000.jpg",
            "images/wrist/demo_wrist_000003.jpg",
        ]

        legacy_project_file = temp_dir / "legacy_project.json"
        legacy_project_file.write_text(
            json.dumps(
                {
                    "version": 2,
                    "video_path": "demo.mp4",
                    "labels": ["待机", "执行中", "异常"],
                    "annotations": {"3": "执行中"},
                    "current_frame_index": 3,
                    "frame_step": 1,
                    "keyframe_threshold": 12.0,
                    "instruction_template": "请判断状态",
                    "input_template": "<image>",
                    "dataset_name": "legacy_demo",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        legacy_project = load_project(legacy_project_file)
        assert legacy_project.label_group_name is None

        existing_project_path = temp_dir / "demo_project.json"
        existing_project_path.write_text(
            json.dumps(
                {
                    "version": 4,
                    "video_path": "demo_top.mp4",
                    "wrist_video_path": "demo_wrist.mp4",
                    "labels": ["未开始", "进行中", "已完成", "失败"],
                    "annotations": {"3": "进行中"},
                    "current_frame_index": 3,
                    "frame_step": 1,
                    "keyframe_threshold": 12.0,
                    "label_group_name": "默认状态标签",
                    "instruction_template": "请根据图片判断当前任务状态，只输出标签名。",
                    "input_template": DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
                    "dataset_name": "demo_project",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        original_project_text = existing_project_path.read_text(encoding="utf-8")

        session = GradioLabelingSession()
        session.load_video(
            str(video_path),
            str(wrist_video_path),
            "",
            str(dataset_dir),
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        assert existing_project_path.read_text(encoding="utf-8") == original_project_text

        mismatch_session = GradioLabelingSession()
        mismatch_session.load_video(
            str(mismatch_top_video_path),
            str(mismatch_wrist_video_path),
            "",
            str(temp_dir / "mismatch_workspace"),
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        mismatch_snapshot = mismatch_session.snapshot()
        assert mismatch_snapshot.frame_count == 12
        assert mismatch_snapshot.top_frame_image is not None
        assert mismatch_snapshot.wrist_frame_image is not None
        assert "共同帧数拼接为 12 帧同步时间轴" in mismatch_snapshot.status_message
        assert "原始帧数" in mismatch_snapshot.frame_info_markdown

        directory_session = GradioLabelingSession()
        directory_session.load_video(
            str(top_dir),
            str(wrist_dir),
            "",
            str(temp_dir / "dir_workspace"),
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        directory_snapshot = directory_session.snapshot()
        assert directory_snapshot.top_input_path == str(top_dir)
        assert directory_snapshot.wrist_input_path == str(wrist_dir)
        assert directory_snapshot.selected_relative_path == "全部视频（2 段）"
        assert directory_snapshot.available_relative_paths == [
            "chunk-000/file-000.mp4",
            "chunk-000/file-001.mp4",
        ]
        assert directory_snapshot.resolved_top_video_path.endswith("chunk-000/file-000.mp4")
        assert directory_snapshot.frame_count == 18
        assert "已拼接 2 段" in directory_snapshot.status_message

        directory_session.load_video(
            str(top_dir),
            str(wrist_dir),
            "chunk-000/file-001.mp4",
            str(temp_dir / "dir_workspace_2"),
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        directory_snapshot = directory_session.snapshot()
        assert directory_snapshot.selected_relative_path == "chunk-000/file-001.mp4"
        assert directory_snapshot.resolved_wrist_video_path.endswith("chunk-000/file-001.mp4")

        directory_session.jump_to_seconds(0.7)
        jumped_snapshot = directory_session.snapshot()
        assert jumped_snapshot.current_seconds == 0.7
        assert "#t=0.700" in jumped_snapshot.sync_player_html
        directory_session.jump_relative_seconds(99.0)
        clamped_snapshot = directory_session.snapshot()
        last_seconds = directory_session.controller.reader.timestamp_seconds(
            directory_session.controller.frame_count - 1
        )
        assert round(clamped_snapshot.current_seconds, 3) == round(last_seconds, 3)
        assert f"{last_seconds:.2f}s" in clamped_snapshot.status_message
        assert "99.0s" in clamped_snapshot.status_message
        assert "99.70s" not in clamped_snapshot.status_message

        lerobot_session = GradioLabelingSession()
        lerobot_session.load_video(
            str(lerobot_root),
            "",
            "chunk-000/file-000.mp4",
            str(temp_dir / "demo_lerobot_export"),
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        lerobot_snapshot = lerobot_session.snapshot()
        assert lerobot_snapshot.top_input_path == str(lerobot_root)
        assert lerobot_snapshot.wrist_input_path == ""
        assert lerobot_snapshot.selected_relative_path == "chunk-000/file-000.mp4"
        assert lerobot_snapshot.available_relative_paths == ["chunk-000/file-000.mp4"]
        assert lerobot_snapshot.resolved_top_video_path.endswith("chunk-000/file-000.mp4")
        assert lerobot_snapshot.resolved_wrist_video_path.endswith("chunk-000/file-000.mp4")
        assert lerobot_snapshot.export_dir.endswith("demo_lerobot_export")
        assert "LeRobot 数据集目录自动解析出 top/wrist 双视角" in lerobot_snapshot.status_message

        lerobot_session.apply_range_label_seconds(0.0, 0.5, "left_hand_pickup")
        lerobot_snapshot = lerobot_session.snapshot()
        lerobot_session.export_dataset(
            lerobot_snapshot.export_dir,
            lerobot_snapshot.dataset_name,
            lerobot_snapshot.instruction_template,
            lerobot_snapshot.input_template,
        )
        exported_snapshot = lerobot_session.snapshot()
        assert exported_snapshot.export_dir.endswith("demo_lerobot_export")
        assert "训练命令" in exported_snapshot.status_message
        assert "run_qwen25vl_lora_train.sh" in exported_snapshot.status_message
        assert "train_qwen25vl_lora.yaml" in exported_snapshot.status_message

        tolerant_lerobot_session = GradioLabelingSession()
        tolerant_lerobot_session.load_video(
            str(lerobot_root),
            str(lerobot_root),
            "chunk-000/file-000.mp4",
            "",
            "demo labels",
            "请根据图片判断当前任务状态，只输出标签名。",
            DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        )
        tolerant_snapshot = tolerant_lerobot_session.snapshot()
        assert tolerant_snapshot.resolved_top_video_path.endswith("chunk-000/file-000.mp4")
        assert tolerant_snapshot.resolved_wrist_video_path.endswith("chunk-000/file-000.mp4")

        sublabel_root = temp_dir / "sublabel_lerobot"
        sublabel_top_dir = sublabel_root / "videos" / "observation.images.top" / "chunk-000"
        sublabel_wrist_dir = sublabel_root / "videos" / "observation.images.right_wrist" / "chunk-000"
        sublabel_left_dir = sublabel_root / "videos" / "observation.images.left_wrist" / "chunk-000"
        for directory in (sublabel_top_dir, sublabel_wrist_dir, sublabel_left_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for file_index, frame_count in enumerate((10, 8)):
            _write_demo_video(sublabel_top_dir / f"file-{file_index:03d}.mp4", frame_count=frame_count)
            _write_demo_video(sublabel_wrist_dir / f"file-{file_index:03d}.mp4", frame_count=frame_count)
            _write_demo_video(sublabel_left_dir / f"file-{file_index:03d}.mp4", frame_count=frame_count)

        sublabel_session = GradioLabelingSession()
        sublabel_session.load_video(
            str(sublabel_root),
            "",
            "",
            str(temp_dir / "sublabel_export"),
            "sublabel demo",
            "请根据图片判断当前任务状态，只输出标签名。",
            "",
        )
        sublabel_snapshot = sublabel_session.snapshot()
        assert sublabel_snapshot.frame_count == 18
        assert sublabel_snapshot.total_seconds == 1.8
        assert sublabel_snapshot.selected_relative_path == "全部视频（2 段）"
        assert sublabel_snapshot.resolved_left_wrist_video_path.endswith("chunk-000/file-000.mp4")

        sublabel_session.apply_range_label_seconds(0.2, 1.4, "left_hand_pickup")
        assert (sublabel_root / "sublabels.json").exists()
        saved_sublabels = json.loads((sublabel_root / "sublabels.json").read_text(encoding="utf-8"))
        assert len(saved_sublabels["sublabels"]) == 1
        assert saved_sublabels["sublabels"][0]["label"] == "left_hand_pickup"
        assert saved_sublabels["total_seconds"] == 1.8

        resumed_session = GradioLabelingSession()
        resumed_session.load_video(
            str(sublabel_root),
            "",
            "",
            str(temp_dir / "sublabel_export_2"),
            "sublabel demo",
            "请根据图片判断当前任务状态，只输出标签名。",
            "",
        )
        assert resumed_session.controller.project.sublabel_count() == 1

        resumed_snapshot = resumed_session.snapshot()
        resumed_session.export_dataset(
            resumed_snapshot.export_dir,
            resumed_snapshot.dataset_name,
            resumed_snapshot.instruction_template,
            resumed_snapshot.input_template,
        )
        exported_records = json.loads(
            (Path(resumed_snapshot.export_dir) / "dataset.json").read_text(encoding="utf-8")
        )
        assert len(exported_records) == 13
        assert all(record["output"] == "left_hand_pickup" for record in exported_records)
        assert all(len(record["images"]) == 3 for record in exported_records)

        dataset_only_dir = temp_dir / "dataset_only"
        dataset_only_dir.mkdir(parents=True, exist_ok=True)
        (dataset_only_dir / "dataset.json").write_text("[]", encoding="utf-8")
        assert not workspace_has_state(dataset_only_dir)

    print("video_labeler workspace smoke test passed.")


if __name__ == "__main__":
    main()
