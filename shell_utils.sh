#!/usr/bin/env bash

# 用法:
#   source /workspace/video_labeler/shell_utils.sh
#
# 作用:
#   提供项目脚本共用的 conda 环境激活逻辑。
#
# 默认配置:
#   优先环境名: video_labeler
#   兼容环境名: qwen25vl_factory_qt, qwen25vl_factory

set -euo pipefail

VIDEO_LABELER_ENV_NAMES=(
  "video_labeler"
  "qwen25vl_factory_qt"
  "qwen25vl_factory"
)

conda_env_exists() {
  local env_name="$1"
  conda env list | grep -Eq "^${env_name}[[:space:]]"
}

current_env_supported() {
  local env_name=""

  if [[ -v CONDA_DEFAULT_ENV ]] && [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    env_name="$CONDA_DEFAULT_ENV"
  fi

  if [[ -z "$env_name" ]]; then
    return 1
  fi

  for candidate in "${VIDEO_LABELER_ENV_NAMES[@]}"; do
    if [[ "$env_name" == "$candidate" ]]; then
      return 0
    fi
  done

  return 1
}

activate_video_labeler_env() {
  source /opt/miniconda3/etc/profile.d/conda.sh

  if current_env_supported; then
    return 0
  fi

  for env_name in "${VIDEO_LABELER_ENV_NAMES[@]}"; do
    if conda_env_exists "$env_name"; then
      if [[ "$env_name" != "${VIDEO_LABELER_ENV_NAMES[0]}" ]]; then
        echo "检测到可兼容的既有环境，继续使用。"
      fi
      conda activate "$env_name"
      return 0
    fi
  done

  echo "未找到可用 conda 环境: ${VIDEO_LABELER_ENV_NAMES[*]}" >&2
  echo "请先运行: bash /workspace/video_labeler/setup_video_labeler.sh" >&2
  return 1
}
