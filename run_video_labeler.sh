#!/usr/bin/env bash

# 用法:
#   bash /workspace/video_labeler/run_video_labeler.sh
#   bash /workspace/video_labeler/run_video_labeler.sh /path/to/top.mp4
#   bash /workspace/video_labeler/run_video_labeler.sh /path/to/top.mp4 /path/to/wrist.mp4
#   bash /workspace/video_labeler/run_video_labeler.sh /path/to/lerobot_dataset_root chunk-000/file-000.mp4
#   bash /workspace/video_labeler/run_video_labeler.sh /path/to/top_dir /path/to/wrist_dir chunk-000/file-000.mp4
#
# 作用:
#   在视频标注环境中启动 Gradio 浏览器版视频逐帧标注工具。
#   适合 VSCode Remote、SSH 和端口转发场景使用。
#
# 默认配置:
#   优先 conda 环境: video_labeler
#   兼容环境名: qwen25vl_factory_qt, qwen25vl_factory
#   默认 LeRobot 数据集目录: /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
#   默认 wrist 输入: 留空，由程序自动解析
#   默认目录内视频: 留空，由 episode 元数据自动加载
#   默认监听地址: 0.0.0.0:7860
#   默认端口搜索范围: 7860-7890
#   默认行为: 不带参数时自动使用内置的 LeRobot 数据集目录

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
DEFAULT_TOP_VIDEO_PATH="/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376"
DEFAULT_WRIST_VIDEO_PATH=""
DEFAULT_RELATIVE_VIDEO_PATH=""

if (( $# == 0 )); then
  TOP_VIDEO_PATH="$DEFAULT_TOP_VIDEO_PATH"
  WRIST_VIDEO_PATH="$DEFAULT_WRIST_VIDEO_PATH"
  RELATIVE_VIDEO_PATH="$DEFAULT_RELATIVE_VIDEO_PATH"
elif (( $# >= 1 )) && [[ -n "$1" ]]; then
  TOP_VIDEO_PATH="$1"
  if (( $# == 2 )) && [[ -n "$2" ]] && [[ "$2" != /* ]] && [[ "$2" == *"/"* || "$2" == *.mp4 ]]; then
    WRIST_VIDEO_PATH=""
    RELATIVE_VIDEO_PATH="$2"
  elif (( $# >= 2 )) && [[ -n "$2" ]]; then
    WRIST_VIDEO_PATH="$2"
    if (( $# >= 3 )) && [[ -n "$3" ]]; then
      RELATIVE_VIDEO_PATH="$3"
    else
      RELATIVE_VIDEO_PATH=""
    fi
  else
    WRIST_VIDEO_PATH=""
    RELATIVE_VIDEO_PATH=""
  fi
else
  TOP_VIDEO_PATH=""
  WRIST_VIDEO_PATH=""
  RELATIVE_VIDEO_PATH=""
fi

source "${SCRIPT_DIR}/shell_utils.sh"
activate_video_labeler_env

if [[ -v HOST ]] && [[ -n "$HOST" ]]; then
  HOST_VALUE="$HOST"
else
  HOST_VALUE="0.0.0.0"
fi

if [[ -v PORT ]] && [[ -n "$PORT" ]]; then
  PORT_VALUE="$PORT"
else
  PORT_VALUE="7860"
fi

if [[ -v PORT_MAX ]] && [[ -n "$PORT_MAX" ]]; then
  PORT_MAX_VALUE="$PORT_MAX"
else
  PORT_MAX_VALUE="7890"
fi

find_available_port() {
  local preferred_port="$1"
  local max_port="$2"

  python - "$preferred_port" "$max_port" <<'PY'
import socket
import sys

preferred_port = int(sys.argv[1])
max_port = int(sys.argv[2])

for port in range(preferred_port, max_port + 1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

if ! SELECTED_PORT="$(find_available_port "$PORT_VALUE" "$PORT_MAX_VALUE")"; then
  echo "未找到可用端口，已尝试范围: ${PORT_VALUE}-${PORT_MAX_VALUE}" >&2
  exit 1
fi

if [[ "$SELECTED_PORT" != "$PORT_VALUE" ]]; then
  echo "端口 ${PORT_VALUE} 已被占用，自动切换到 ${SELECTED_PORT}。"
fi

echo "Gradio 服务即将启动:"
echo "  监听地址: http://${HOST_VALUE}:${SELECTED_PORT}"
echo "  本机访问: http://127.0.0.1:${SELECTED_PORT}"
if [[ -n "$TOP_VIDEO_PATH" ]]; then
  echo "  top 视频: ${TOP_VIDEO_PATH}"
fi
if [[ -n "$WRIST_VIDEO_PATH" ]]; then
  echo "  wrist 视频: ${WRIST_VIDEO_PATH}"
fi
if [[ -n "$RELATIVE_VIDEO_PATH" ]]; then
  echo "  目录内视频: ${RELATIVE_VIDEO_PATH}"
fi

if [[ -n "$TOP_VIDEO_PATH" ]]; then
  if [[ -n "$WRIST_VIDEO_PATH" ]]; then
    if [[ -n "$RELATIVE_VIDEO_PATH" ]]; then
      PYTHONPATH="$PACKAGE_PARENT" python -m video_labeler.main_gradio \
        --host "$HOST_VALUE" \
        --port "$SELECTED_PORT" \
        --top-video-path "$TOP_VIDEO_PATH" \
        --wrist-video-path "$WRIST_VIDEO_PATH" \
        --relative-video-path "$RELATIVE_VIDEO_PATH"
    else
      PYTHONPATH="$PACKAGE_PARENT" python -m video_labeler.main_gradio \
        --host "$HOST_VALUE" \
        --port "$SELECTED_PORT" \
        --top-video-path "$TOP_VIDEO_PATH" \
        --wrist-video-path "$WRIST_VIDEO_PATH"
    fi
  else
    if [[ -n "$RELATIVE_VIDEO_PATH" ]]; then
      PYTHONPATH="$PACKAGE_PARENT" python -m video_labeler.main_gradio \
        --host "$HOST_VALUE" \
        --port "$SELECTED_PORT" \
        --top-video-path "$TOP_VIDEO_PATH" \
        --relative-video-path "$RELATIVE_VIDEO_PATH"
    else
      PYTHONPATH="$PACKAGE_PARENT" python -m video_labeler.main_gradio \
        --host "$HOST_VALUE" \
        --port "$SELECTED_PORT" \
        --top-video-path "$TOP_VIDEO_PATH"
    fi
  fi
else
  PYTHONPATH="$PACKAGE_PARENT" python -m video_labeler.main_gradio \
    --host "$HOST_VALUE" \
    --port "$SELECTED_PORT"
fi
