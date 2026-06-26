"""Script-style smoke tests for dataset record generation."""

from __future__ import annotations

from pathlib import Path

from video_labeler.exporter import (
    ExportSample,
    build_dataset_info,
    build_dataset_record,
    build_export_report,
    build_split_dataset_info,
    sanitize_dataset_name,
    split_export_samples,
)
from video_labeler.models import DEFAULT_DUAL_VIEW_INPUT_TEMPLATE, ProjectState
from video_labeler.training_config import (
    TRAINING_CONFIG_FILE_NAME,
    TRAINING_README_FILE_NAME,
    TrainingConfigError,
    build_training_readme,
    build_training_yaml,
    select_dataset_keys,
)


def main() -> None:
    """Run a small set of pure-Python export assertions."""
    project = ProjectState(
        video_path=Path("/workspace/videos/demo.mp4"),
        wrist_video_path=Path("/workspace/videos/demo_wrist.mp4"),
        input_template=DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
        dataset_name="task state labels",
    )
    record = build_dataset_record(
        project,
        frame_index=12,
        label="进行中",
        image_relative_paths=[
            "images/top/demo_000012.jpg",
            "images/wrist/demo_wrist_000012.jpg",
        ],
        timestamp_seconds=0.5,
    )

    assert record["instruction"] == project.instruction_template
    assert record["output"] == "进行中"
    assert record["images"] == [
        "images/top/demo_000012.jpg",
        "images/wrist/demo_wrist_000012.jpg",
    ]
    assert record["input"] == DEFAULT_DUAL_VIEW_INPUT_TEMPLATE

    dataset_info = build_dataset_info(project.dataset_name)
    assert sanitize_dataset_name(project.dataset_name) in dataset_info
    dataset_key = next(iter(dataset_info))
    assert dataset_info[dataset_key]["columns"]["images"] == "images"

    split_info = build_split_dataset_info(
        project.dataset_name,
        {"train": "train.json", "val": "val.json"},
    )
    assert f"{sanitize_dataset_name(project.dataset_name)}_train" in split_info
    assert f"{sanitize_dataset_name(project.dataset_name)}_val" in split_info

    samples = [
        ExportSample(
            frame_index=index * 10,
            timestamp_seconds=float(index),
            label="进行中" if index % 2 == 0 else "未开始",
            image_relative_paths=[
                f"images/top/demo_{index:06d}.jpg",
                f"images/wrist/demo_wrist_{index:06d}.jpg",
            ],
            record={
                "instruction": "x",
                "input": DEFAULT_DUAL_VIEW_INPUT_TEMPLATE,
                "output": "进行中",
                "images": [],
            },
        )
        for index in range(10)
    ]
    split_samples = split_export_samples(samples)
    assert len(split_samples["train"]) == 8
    assert len(split_samples["val"]) == 1
    assert len(split_samples["test"]) == 1

    report = build_export_report(
        project,
        frame_count=200,
        split_samples=split_samples,
        all_samples=samples,
    )
    assert report["frame_count"] == 200
    assert report["labeled_sample_count"] == 10
    assert report["wrist_video_path"] == str(project.wrist_video_path)
    assert "train" in report["split_summary"]

    split_keys = select_dataset_keys(
        {
            "demo_labels": {"file_name": "dataset.json"},
            "demo_labels_train": {"file_name": "train.json"},
            "demo_labels_val": {"file_name": "val.json"},
            "demo_labels_test": {"file_name": "test.json"},
        }
    )
    assert split_keys.dataset_key == "demo_labels_train"
    assert split_keys.eval_dataset_key == "demo_labels_val"
    assert split_keys.val_size == 0.0

    train_only_split_keys = select_dataset_keys(
        {
            "demo_labels": {"file_name": "dataset.json"},
            "demo_labels_train": {"file_name": "train.json"},
        }
    )
    assert train_only_split_keys.dataset_key == "demo_labels_train"
    assert train_only_split_keys.eval_dataset_key is None
    assert train_only_split_keys.val_size == 0.0

    full_keys = select_dataset_keys(
        {
            "demo_labels": {"file_name": "dataset.json"},
        }
    )
    assert full_keys.dataset_key == "demo_labels"
    assert full_keys.eval_dataset_key is None
    assert full_keys.val_size == 0.05

    try:
        select_dataset_keys({})
    except TrainingConfigError as exc:
        assert "dataset_info" in str(exc)
    else:
        raise AssertionError("Expected invalid dataset_info to fail.")

    training_yaml = build_training_yaml(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
        dataset_key="demo_labels_train",
        eval_dataset_key="demo_labels_val",
        val_size=0.0,
    )
    assert (
        "model_name_or_path: /workspace/vlm_models/models/Qwen2.5-VL-7B-Instruct"
        in training_yaml
    )
    assert "dataset_dir: /workspace/dataset/qwen_finetune/demo_labels" in training_yaml
    assert "dataset: demo_labels_train" in training_yaml
    assert "eval_dataset: demo_labels_val" in training_yaml
    assert "val_size: 0.0" in training_yaml
    assert (
        "output_dir: /workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"
        in training_yaml
    )

    no_eval_yaml = build_training_yaml(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
        dataset_key="demo_labels",
        eval_dataset_key=None,
        val_size=0.05,
    )
    assert "dataset: demo_labels\n" in no_eval_yaml
    assert "eval_dataset:" not in no_eval_yaml
    assert "val_size: 0.05" in no_eval_yaml
    assert "eval_strategy: steps" in no_eval_yaml

    train_only_no_eval_yaml = build_training_yaml(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
        dataset_key="demo_labels_train",
        eval_dataset_key=None,
        val_size=0.0,
    )
    assert "dataset: demo_labels_train\n" in train_only_no_eval_yaml
    assert "eval_dataset:" not in train_only_no_eval_yaml
    assert "val_size: 0.0" in train_only_no_eval_yaml
    assert "eval_strategy: steps" not in train_only_no_eval_yaml
    assert "per_device_eval_batch_size" not in train_only_no_eval_yaml

    readme = build_training_readme(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        config_path=Path("/workspace/dataset/qwen_finetune/demo_labels")
        / TRAINING_CONFIG_FILE_NAME,
        command=(
            "bash /workspace/video_finetune/run_qwen25vl_lora_train.sh "
            "/workspace/dataset/qwen_finetune/demo_labels/train_qwen25vl_lora.yaml"
        ),
        dataset_key="demo_labels_train",
        eval_dataset_key="demo_labels_val",
        val_size=0.0,
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
    )
    assert TRAINING_CONFIG_FILE_NAME in readme
    assert TRAINING_README_FILE_NAME == "training_readme.md"
    assert "demo_labels_train" in readme
    assert "demo_labels_val" in readme
    assert "run_qwen25vl_lora_train.sh" in readme

    disabled_eval_readme = build_training_readme(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        config_path=Path("/workspace/dataset/qwen_finetune/demo_labels")
        / TRAINING_CONFIG_FILE_NAME,
        command=(
            "bash /workspace/video_finetune/run_qwen25vl_lora_train.sh "
            "/workspace/dataset/qwen_finetune/demo_labels/train_qwen25vl_lora.yaml"
        ),
        dataset_key="demo_labels_train",
        eval_dataset_key=None,
        val_size=0.0,
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
    )
    assert "Eval dataset: `disabled`" in disabled_eval_readme
    print("video_labeler exporter smoke test passed.")


if __name__ == "__main__":
    main()
