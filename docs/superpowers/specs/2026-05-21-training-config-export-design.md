# Training Config Export Design

## Goal

After a dataset export, the labeler should leave the export directory ready for
Qwen2.5-VL LoRA training. A user should not need to inspect `dataset_info.json`
or edit a recipe under `/workspace/video_finetune` by hand.

## Scope

This design covers generating a training YAML and a short README beside the
exported dataset. It does not start training from the browser and does not add a
general multi-model template system.

## User Flow

1. The user labels videos and clicks `导出微调数据集`.
2. The labeler writes the normal dataset files: `dataset.json`,
   `dataset_info.json`, split files, `images/`, and report files.
3. The labeler also writes:
   - `train_qwen25vl_lora.yaml`
   - `training_readme.md`
4. The page status shows the command:

   ```bash
   bash /workspace/video_finetune/run_qwen25vl_lora_train.sh /path/to/export/train_qwen25vl_lora.yaml
   ```

## Architecture

Add one module in the labeler package:

- `video_labeler/training_config.py`

Responsibilities:

- Parse the in-memory `dataset_info` payload or the written `dataset_info.json`.
- Select the training dataset key.
- Render a Qwen2.5-VL LoRA YAML.
- Render `training_readme.md`.
- Return generated paths and the runnable training command.

`workspace_store.persist_project_workspace(...)` remains responsible for the
core dataset export. After it writes `dataset_info.json`, it calls the training
config module. `ExportResult` gains optional fields:

- `training_config_file: Path | None`
- `training_readme_file: Path | None`
- `training_command: str | None`
- `training_config_error: str | None`

The Gradio session and app only display the command or the error; they do not
need to know how the YAML is built.

## Dataset Key Selection

The generator reads keys from `dataset_info.json`.

If split keys exist:

- Use `<dataset>_train` as `dataset`.
- Use `<dataset>_val` as `eval_dataset` when present.
- Set `val_size: 0.0` to avoid a second split inside LLaMA-Factory.

If no split keys exist:

- Use the full dataset key as `dataset`.
- Omit `eval_dataset`.
- Set `val_size: 0.05` so LLaMA-Factory can create a small validation split.

If no usable key exists, the dataset export still succeeds, but training config
generation reports a clear error.

## Generated YAML Defaults

The generated YAML targets the current training stack:

- `model_name_or_path: /workspace/vlm_models/models/Qwen2.5-VL-7B-Instruct`
- `template: qwen2_vl`
- `stage: sft`
- `finetuning_type: lora`
- `lora_target: all`
- `lora_rank: 16`
- `lora_alpha: 32`
- `lora_dropout: 0.05`
- `image_max_pixels: 589824`
- `cutoff_len: 4096`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `learning_rate: 1.0e-4`
- `num_train_epochs: 3.0`
- `bf16: true`
- `gradient_checkpointing: true`

The YAML sets:

- `dataset_dir` to the absolute export directory.
- `dataset` to the selected training key.
- `eval_dataset` only when a validation split key exists.
- `output_dir` to
  `/workspace/video_finetune/outputs/<export_dir_name>_qwen25vl_lora`.

Each export overwrites `train_qwen25vl_lora.yaml` and `training_readme.md`
because these files should reflect the latest exported dataset state.

## Error Handling

Training config generation is an export enhancement, not a blocker.

- If dataset export fails, keep existing failure behavior.
- If the exported dataset has zero samples, keep existing validation behavior
  and do not generate training files.
- If training config generation fails after the dataset is written, return a
  successful `ExportResult` with `training_config_error` set.
- The UI status should say that the dataset was exported and include the
  training-config error.

## Testing

Add focused pure-Python tests for `training_config.py`:

- Split dataset keys choose `dataset=<name>_train`,
  `eval_dataset=<name>_val`, and `val_size: 0.0`.
- Full dataset only chooses `dataset=<name>` and `val_size: 0.05`.
- Invalid `dataset_info` produces a clear error.

Extend the workspace export smoke test:

- Assert `train_qwen25vl_lora.yaml` exists.
- Assert `training_readme.md` exists.
- Assert the YAML contains the current `dataset_dir`.
- Assert the selected `dataset`, optional `eval_dataset`, and `output_dir` match
  the export directory and generated `dataset_info.json`.

Also update the existing range-label assertion to match current 1-second
sampling behavior, so the smoke test can be trusted during this work.

## Success Criteria

After exporting any non-empty dataset from the labeler, the export directory
contains a matching training YAML and README. The browser status shows one
command that can be pasted into a terminal to start Qwen2.5-VL LoRA training on
the newly exported dataset.
