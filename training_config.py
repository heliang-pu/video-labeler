"""Generate Qwen2.5-VL LoRA training files for exported datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    full_keys = sorted(
        key
        for key in keys
        if not any(key.endswith(f"_{suffix}") for suffix in ("train", "val", "test"))
    )
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
    should_eval = eval_dataset_key is not None or val_size > 1e-6
    eval_block = (
        "### eval\n"
        f"val_size: {val_size}\n"
        "per_device_eval_batch_size: 1\n"
        "eval_strategy: steps\n"
        "eval_steps: 200\n"
        if should_eval
        else "### eval\nval_size: 0.0\n"
    )
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
        f"{eval_block}"
    )


def build_training_readme(
    *,
    dataset_dir: Path,
    config_path: Path,
    command: str,
    dataset_key: str,
    eval_dataset_key: str | None,
    val_size: float,
    output_dir: Path,
) -> str:
    """Render a short README explaining how to train this exported dataset."""
    if eval_dataset_key is not None:
        eval_text = eval_dataset_key
    elif val_size > 1e-6:
        eval_text = "LLaMA-Factory val_size split"
    else:
        eval_text = "disabled"
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
        f"bash {TRAINING_SCRIPT_PATH.as_posix()} " f"{training_config_path.as_posix()}"
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
        val_size=selected_keys.val_size,
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
