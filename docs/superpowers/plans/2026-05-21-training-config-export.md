# Training Config Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a Qwen2.5-VL LoRA training YAML and runnable command beside every successful non-empty dataset export.

**Architecture:** Add a focused `video_labeler.training_config` module that selects dataset keys from `dataset_info.json` and renders `train_qwen25vl_lora.yaml` plus `training_readme.md`. Keep dataset export ownership in `workspace_store.py`; it calls the new module after writing `dataset_info.json` and stores the generated command on `ExportResult`. The Gradio session only displays the command or a generation error.

**Tech Stack:** Python 3.11, dataclasses, pathlib, JSON payloads, plain YAML string rendering, existing script-style smoke tests.

---

## Current Workspace Note

`/workspace/video_labeler` already has uncommitted changes in `gradio_app.py`, `gradio_session.py`, `models.py`, `video.py`, `workspace_store.py`, plus an untracked `episode_metadata.py`. Treat these as existing user/workspace changes. Do not revert them. When committing, stage only the files intentionally changed for this feature.

## File Structure

- Create `video_labeler/training_config.py`
  - Owns dataset-key selection, YAML rendering, README rendering, and writing generated training files.
  - Public API: `generate_training_files(output_dir: Path, dataset_info: dict[str, object]) -> TrainingConfigResult`.
- Modify `video_labeler/exporter.py`
  - Extend `ExportResult` with optional training config fields.
- Modify `video_labeler/workspace_store.py`
  - Import and call `generate_training_files` after writing `dataset_info.json`.
  - Preserve successful dataset export if training config generation fails.
- Modify `video_labeler/gradio_session.py`
  - Add training command or training config error to export success status.
- Modify `video_labeler/test_video_labeler_export.py`
  - Add pure-Python tests for dataset key selection and YAML/README rendering.
- Modify `video_labeler/test_video_labeler_workspace.py`
  - Assert generated files exist during export.
  - Fix current range-label assertion to match 1-second interval sampling.
- Optional docs update if implementation reveals command text differs:
  - `video_labeler/README.md`

## Task 1: Training Config Pure Functions

**Files:**
- Create: `/workspace/video_labeler/training_config.py`
- Modify: `/workspace/video_labeler/test_video_labeler_export.py`

- [ ] **Step 1: Add failing pure-function tests**

Append these imports in `/workspace/video_labeler/test_video_labeler_export.py`:

```python
from video_labeler.training_config import (
    TRAINING_CONFIG_FILE_NAME,
    TRAINING_README_FILE_NAME,
    TrainingConfigError,
    build_training_readme,
    build_training_yaml,
    select_dataset_keys,
)
```

Then add these assertions inside `main()` before the final `print(...)`:

```python
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
    assert "model_name_or_path: /workspace/vlm_models/models/Qwen2.5-VL-7B-Instruct" in training_yaml
    assert "dataset_dir: /workspace/dataset/qwen_finetune/demo_labels" in training_yaml
    assert "dataset: demo_labels_train" in training_yaml
    assert "eval_dataset: demo_labels_val" in training_yaml
    assert "val_size: 0.0" in training_yaml
    assert "output_dir: /workspace/video_finetune/outputs/demo_labels_qwen25vl_lora" in training_yaml

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

    readme = build_training_readme(
        dataset_dir=Path("/workspace/dataset/qwen_finetune/demo_labels"),
        config_path=Path("/workspace/dataset/qwen_finetune/demo_labels") / TRAINING_CONFIG_FILE_NAME,
        command=(
            "bash /workspace/video_finetune/run_qwen25vl_lora_train.sh "
            "/workspace/dataset/qwen_finetune/demo_labels/train_qwen25vl_lora.yaml"
        ),
        dataset_key="demo_labels_train",
        eval_dataset_key="demo_labels_val",
        output_dir=Path("/workspace/video_finetune/outputs/demo_labels_qwen25vl_lora"),
    )
    assert TRAINING_CONFIG_FILE_NAME in readme
    assert TRAINING_README_FILE_NAME == "training_readme.md"
    assert "demo_labels_train" in readme
    assert "demo_labels_val" in readme
    assert "run_qwen25vl_lora_train.sh" in readme
```

- [ ] **Step 2: Run the test and verify it fails for the missing module**

Run:

```bash
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'video_labeler.training_config'`.

- [ ] **Step 3: Create `training_config.py`**

Create `/workspace/video_labeler/training_config.py`:

```python
"""Generate Qwen2.5-VL LoRA training files for exported datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAINING_CONFIG_FILE_NAME = "train_qwen25vl_lora.yaml"
TRAINING_README_FILE_NAME = "training_readme.md"
DEFAULT_MODEL_PATH = Path("/workspace/vlm_models/models/Qwen2.5-VL-7B-Instruct")
DEFAULT_OUTPUT_ROOT = Path("/workspace/video_finetune/outputs")
TRAINING_SCRIPT_PATH = Path("/workspace/video_finetune/run_qwen25vl_lora_train.sh")


class TrainingConfigError(RuntimeError):
    """Raised when a dataset_info payload cannot produce a training config."""


@dataclass(slots=True)
class SelectedDatasetKeys:
    """Dataset keys selected for LLaMA-Factory training."""

    dataset_key: str
    eval_dataset_key: str | None
    val_size: float


@dataclass(slots=True)
class TrainingConfigResult:
    """Generated training files and command for one dataset export."""

    config_file: Path
    readme_file: Path
    command: str
    dataset_key: str
    eval_dataset_key: str | None
    output_dir: Path


def select_dataset_keys(dataset_info: dict[str, object]) -> SelectedDatasetKeys:
    """Select train/eval dataset keys from one LLaMA-Factory dataset_info payload."""
    keys = [
        key
        for key, value in dataset_info.items()
        if isinstance(key, str) and key.strip() and isinstance(value, dict)
    ]
    if not keys:
        raise TrainingConfigError("dataset_info does not contain any usable dataset entries.")

    split_train_keys = sorted(key for key in keys if key.endswith("_train"))
    if split_train_keys:
        train_key = split_train_keys[0]
        base_key = train_key[: -len("_train")]
        val_key = f"{base_key}_val"
        return SelectedDatasetKeys(
            dataset_key=train_key,
            eval_dataset_key=val_key if val_key in keys else None,
            val_size=0.0,
        )

    full_keys = sorted(key for key in keys if not any(key.endswith(f"_{suffix}") for suffix in ("train", "val", "test")))
    if full_keys:
        return SelectedDatasetKeys(
            dataset_key=full_keys[0],
            eval_dataset_key=None,
            val_size=0.05,
        )

    raise TrainingConfigError("dataset_info only contains split entries without a *_train key.")


def build_training_yaml(
    *,
    dataset_dir: Path,
    output_dir: Path,
    dataset_key: str,
    eval_dataset_key: str | None,
    val_size: float,
) -> str:
    """Render a Qwen2.5-VL LoRA YAML for LLaMA-Factory."""
    eval_line = "" if eval_dataset_key is None else f"eval_dataset: {eval_dataset_key}\n"
    return (
        "### model\n"
        f"model_name_or_path: {DEFAULT_MODEL_PATH.as_posix()}\n"
        "trust_remote_code: true\n"
        "image_max_pixels: 589824\n"
        "\n"
        "### method\n"
        "stage: sft\n"
        "do_train: true\n"
        "finetuning_type: lora\n"
        "lora_target: all\n"
        "lora_rank: 16\n"
        "lora_alpha: 32\n"
        "lora_dropout: 0.05\n"
        "\n"
        "### dataset\n"
        f"dataset_dir: {dataset_dir.as_posix()}\n"
        f"dataset: {dataset_key}\n"
        f"{eval_line}"
        "template: qwen2_vl\n"
        "cutoff_len: 4096\n"
        "overwrite_cache: true\n"
        "preprocessing_num_workers: 8\n"
        "dataloader_num_workers: 4\n"
        "\n"
        "### output\n"
        f"output_dir: {output_dir.as_posix()}\n"
        "logging_steps: 10\n"
        "save_steps: 200\n"
        "save_total_limit: 2\n"
        "plot_loss: true\n"
        "overwrite_output_dir: false\n"
        "report_to: none\n"
        "\n"
        "### train\n"
        "per_device_train_batch_size: 1\n"
        "gradient_accumulation_steps: 8\n"
        "learning_rate: 1.0e-4\n"
        "num_train_epochs: 3.0\n"
        "lr_scheduler_type: cosine\n"
        "warmup_ratio: 0.1\n"
        "bf16: true\n"
        "gradient_checkpointing: true\n"
        "ddp_timeout: 180000000\n"
        "\n"
        "### eval\n"
        f"val_size: {val_size}\n"
        "per_device_eval_batch_size: 1\n"
        "eval_strategy: steps\n"
        "eval_steps: 200\n"
    )


def build_training_readme(
    *,
    dataset_dir: Path,
    config_path: Path,
    command: str,
    dataset_key: str,
    eval_dataset_key: str | None,
    output_dir: Path,
) -> str:
    """Render a short README explaining how to train this exported dataset."""
    eval_text = eval_dataset_key if eval_dataset_key is not None else "LLaMA-Factory val_size split"
    return (
        "# Training This Dataset\n"
        "\n"
        "This directory contains a LLaMA-Factory multimodal dataset exported by video_labeler.\n"
        "\n"
        "## Files\n"
        "\n"
        f"- Dataset directory: `{dataset_dir.as_posix()}`\n"
        f"- Training config: `{config_path.as_posix()}`\n"
        f"- Dataset key: `{dataset_key}`\n"
        f"- Eval dataset: `{eval_text}`\n"
        f"- Output directory: `{output_dir.as_posix()}`\n"
        "\n"
        "## Train\n"
        "\n"
        "```bash\n"
        f"{command}\n"
        "```\n"
    )


def generate_training_files(
    *,
    output_dir: Path,
    dataset_info: dict[str, object],
) -> TrainingConfigResult:
    """Write training YAML and README beside one exported dataset."""
    resolved_output_dir = output_dir.expanduser().resolve()
    selected_keys = select_dataset_keys(dataset_info)
    training_config_path = resolved_output_dir / TRAINING_CONFIG_FILE_NAME
    training_readme_path = resolved_output_dir / TRAINING_README_FILE_NAME
    training_output_dir = DEFAULT_OUTPUT_ROOT / f"{resolved_output_dir.name}_qwen25vl_lora"
    command = (
        f"bash {TRAINING_SCRIPT_PATH.as_posix()} "
        f"{training_config_path.as_posix()}"
    )
    training_yaml = build_training_yaml(
        dataset_dir=resolved_output_dir,
        output_dir=training_output_dir,
        dataset_key=selected_keys.dataset_key,
        eval_dataset_key=selected_keys.eval_dataset_key,
        val_size=selected_keys.val_size,
    )
    training_readme = build_training_readme(
        dataset_dir=resolved_output_dir,
        config_path=training_config_path,
        command=command,
        dataset_key=selected_keys.dataset_key,
        eval_dataset_key=selected_keys.eval_dataset_key,
        output_dir=training_output_dir,
    )
    training_config_path.write_text(training_yaml, encoding="utf-8")
    training_readme_path.write_text(training_readme, encoding="utf-8")
    return TrainingConfigResult(
        config_file=training_config_path,
        readme_file=training_readme_path,
        command=command,
        dataset_key=selected_keys.dataset_key,
        eval_dataset_key=selected_keys.eval_dataset_key,
        output_dir=training_output_dir,
    )
```

- [ ] **Step 4: Run the pure-function test**

Run:

```bash
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
```

Expected: PASS and print `video_labeler exporter smoke test passed.`

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git -C /workspace/video_labeler status --short
git -C /workspace/video_labeler add training_config.py test_video_labeler_export.py
git -C /workspace/video_labeler commit -m "labeler: add training config generator"
```

Expected: commit succeeds. The status may still show pre-existing unrelated modified files; do not stage them.

## Task 2: Integrate Training Files Into Dataset Export

**Files:**
- Modify: `/workspace/video_labeler/exporter.py`
- Modify: `/workspace/video_labeler/workspace_store.py`
- Modify: `/workspace/video_labeler/test_video_labeler_workspace.py`

- [ ] **Step 1: Add failing workspace assertions for generated training files**

In `/workspace/video_labeler/test_video_labeler_workspace.py`, after the existing first export assertions:

```python
        assert (dataset_dir / "train_qwen25vl_lora.yaml").exists()
        assert (dataset_dir / "training_readme.md").exists()
        assert first_result.training_config_file == dataset_dir / "train_qwen25vl_lora.yaml"
        assert first_result.training_readme_file == dataset_dir / "training_readme.md"
        assert first_result.training_command == (
            "bash /workspace/video_finetune/run_qwen25vl_lora_train.sh "
            f"{dataset_dir / 'train_qwen25vl_lora.yaml'}"
        )
        assert first_result.training_config_error is None
        training_yaml_text = (dataset_dir / "train_qwen25vl_lora.yaml").read_text(encoding="utf-8")
        assert f"dataset_dir: {dataset_dir}" in training_yaml_text
        assert "dataset: demo_labels_train" in training_yaml_text
        assert "eval_dataset: demo_labels_val" in training_yaml_text
        assert "val_size: 0.0" in training_yaml_text
        assert "output_dir: /workspace/video_finetune/outputs/demo_workspace_qwen25vl_lora" in training_yaml_text
```

Place it after:

```python
        assert (dataset_dir / "test.json").exists()
```

For the second small export, add these assertions after `assert second_result.exported_samples == 2`:

```python
        second_yaml_text = (dataset_dir / "train_qwen25vl_lora.yaml").read_text(encoding="utf-8")
        assert "dataset: demo_labels_train" in second_yaml_text
        assert "eval_dataset:" not in second_yaml_text
        assert "val_size: 0.0" in second_yaml_text
```

- [ ] **Step 2: Run the workspace test and verify it fails**

Run in the conda environment that has OpenCV:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected: FAIL because `ExportResult` does not yet have `training_config_file`, or because generated files do not exist.

- [ ] **Step 3: Extend `ExportResult`**

Modify `/workspace/video_labeler/exporter.py` so `ExportResult` includes these fields:

```python
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
```

- [ ] **Step 4: Call the generator from `workspace_store.py`**

At the top of `/workspace/video_labeler/workspace_store.py`, add:

```python
from .training_config import generate_training_files
```

Inside `persist_project_workspace(...)`, just before writing `dataset_info_file`, replace the direct call with a named payload:

```python
        dataset_info_payload = build_split_dataset_info(project.dataset_name, split_file_names)
        _write_json_atomic(
            dataset_info_file,
            dataset_info_payload,
        )
```

Before the `try:` block, initialize:

```python
    training_config_file = None
    training_readme_file = None
    training_command = None
    training_config_error = None
```

After writing `report_file`, add:

```python
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
```

Update the final `ExportResult(...)` call:

```python
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
```

- [ ] **Step 5: Run the workspace test**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected: It may still fail at the existing range-label assertion near the end (`assert len(exported_records) == 1`). If it fails there, continue to Step 6.

- [ ] **Step 6: Fix the existing range-label assertion**

In `/workspace/video_labeler/test_video_labeler_workspace.py`, replace:

```python
        assert len(exported_records) == 1
        assert exported_records[0]["output"] == "进行中"
        assert len(exported_records[0]["images"]) == 3
```

with:

```python
        assert len(exported_records) == 2
        assert all(record["output"] == "进行中" for record in exported_records)
        assert all(len(record["images"]) == 3 for record in exported_records)
```

Rationale: `_build_sublabel_samples(...)` samples from `start_seconds` to `end_seconds` in 1-second increments. The test labels `0.2` to `1.4`, so it exports samples at `0.2` and `1.2`.

- [ ] **Step 7: Run workspace and export tests**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected:

- `video_labeler exporter smoke test passed.`
- `video_labeler workspace smoke test passed.`

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git -C /workspace/video_labeler status --short
git -C /workspace/video_labeler add exporter.py workspace_store.py test_video_labeler_workspace.py
git -C /workspace/video_labeler commit -m "labeler: write training files on export"
```

Expected: commit succeeds. Do not stage unrelated existing files unless they were intentionally changed for this task.

## Task 3: Show Training Command in Gradio Export Status

**Files:**
- Modify: `/workspace/video_labeler/gradio_session.py`
- Modify: `/workspace/video_labeler/test_video_labeler_workspace.py`

- [ ] **Step 1: Add a failing status assertion**

In `/workspace/video_labeler/test_video_labeler_workspace.py`, find the section where `lerobot_session.export_dataset(...)` is called. After:

```python
        exported_snapshot = lerobot_session.snapshot()
```

add:

```python
        assert "训练命令" in exported_snapshot.status_message
        assert "run_qwen25vl_lora_train.sh" in exported_snapshot.status_message
        assert "train_qwen25vl_lora.yaml" in exported_snapshot.status_message
```

- [ ] **Step 2: Run the workspace test and verify it fails**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected: FAIL because the status message does not yet include `训练命令`.

- [ ] **Step 3: Update export status in `gradio_session.py`**

In `/workspace/video_labeler/gradio_session.py`, replace the status block after `split_text = ...` with:

```python
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
```

This keeps successful export status even when training config generation fails.

- [ ] **Step 4: Run the workspace test**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected: PASS and print `video_labeler workspace smoke test passed.`

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git -C /workspace/video_labeler status --short
git -C /workspace/video_labeler add gradio_session.py test_video_labeler_workspace.py
git -C /workspace/video_labeler commit -m "labeler: show generated training command"
```

Expected: commit succeeds, staging only the intended files.

## Task 4: Final Verification and Docs Touch-Up

**Files:**
- Modify if needed: `/workspace/video_labeler/README.md`

- [ ] **Step 1: Run the full relevant smoke test set**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
PYTHONPATH=/workspace python -m video_labeler.main_gradio --help
```

Expected:

- Export smoke test prints `video_labeler exporter smoke test passed.`
- Workspace smoke test prints `video_labeler workspace smoke test passed.`
- `main_gradio --help` prints CLI usage and exits 0.

- [ ] **Step 2: Update README if generated files are not documented**

If `/workspace/video_labeler/README.md` does not already mention generated training files, add these bullets under the auto-save/export file list:

```text
- `train_qwen25vl_lora.yaml`
- `training_readme.md`
```

And add this short section near `配套训练项目`:

```text
导出完成后，数据集目录会生成 `train_qwen25vl_lora.yaml` 和 `training_readme.md`。页面状态栏会显示可直接运行的训练命令，例如：

    bash /workspace/video_finetune/run_qwen25vl_lora_train.sh /workspace/dataset/qwen_finetune/<数据集名>/train_qwen25vl_lora.yaml
```

- [ ] **Step 3: Run README-independent tests again if README changed**

Run:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen25vl_factory_qt
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
```

Expected: both tests pass.

- [ ] **Step 4: Commit docs if changed**

If README changed, run:

```bash
git -C /workspace/video_labeler add README.md
git -C /workspace/video_labeler commit -m "docs: describe generated training config"
```

If README did not change, skip this commit.

- [ ] **Step 5: Final status check**

Run:

```bash
git -C /workspace/video_labeler status --short
```

Expected: no uncommitted changes from this feature. Pre-existing unrelated workspace changes may remain; list them in the final handoff.
