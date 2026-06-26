#!/usr/bin/env bash

# 用法:
#   bash /workspace/video_labeler/setup_video_labeler.sh
#
# 作用:
#   创建并配置独立的浏览器版视频标注环境。
#   该环境只安装标注器运行所需的轻量依赖。
#
# 默认配置:
#   conda 环境名: video_labeler
#   Python 版本: 3.11
#   依赖来源: /workspace/video_labeler/requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="video_labeler"

source /opt/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -Eq "^${ENV_NAME}[[:space:]]"; then
  echo "Conda environment already exists: ${ENV_NAME}"
else
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

python - <<'PY'
import cv2
import gradio

print(f"opencv={cv2.__version__}")
print(f"gradio={gradio.__version__}")
PY
