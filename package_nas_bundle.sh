#!/usr/bin/env bash
#
# 用法:
#   bash /workspace/video_labeler/package_nas_bundle.sh
#   bash /workspace/video_labeler/package_nas_bundle.sh video-labeler:latest my_bundle_name
#
# 作用:
#   将 NAS 部署需要的 Docker 镜像、compose 配置、标签配置、源码备份和交接文档打包到 dist/。
#
# 默认配置:
#   镜像: video-labeler:latest
#   输出目录: /workspace/video_labeler/dist/video_labeler_nas_bundle_<时间戳>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ge 1 ] && [ -n "$1" ]; then
  IMAGE_TAG="$1"
else
  IMAGE_TAG="video-labeler:latest"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [ "$#" -ge 2 ] && [ -n "$2" ]; then
  BUNDLE_NAME="$2"
else
  BUNDLE_NAME="video_labeler_nas_bundle_${TIMESTAMP}"
fi

DIST_DIR="${SCRIPT_DIR}/dist"
BUNDLE_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.tar.gz"

mkdir -p "${BUNDLE_DIR}/logs"

docker image inspect "${IMAGE_TAG}" >/dev/null

echo "保存 Docker 镜像: ${IMAGE_TAG}"
docker save "${IMAGE_TAG}" -o "${BUNDLE_DIR}/video-labeler_latest.tar"

echo "复制部署配置和文档"
cp "${SCRIPT_DIR}/docker-compose.nas.yml" "${BUNDLE_DIR}/docker-compose.yml"
cp "${SCRIPT_DIR}/label_groups.json" "${BUNDLE_DIR}/label_groups.json"
cp "${SCRIPT_DIR}/NAS_AGENT_HANDOFF.md" "${BUNDLE_DIR}/NAS_AGENT_HANDOFF.md"
cp "${SCRIPT_DIR}/PROMPT_FOR_NAS_AGENT.md" "${BUNDLE_DIR}/PROMPT_FOR_NAS_AGENT.md"
cp "${SCRIPT_DIR}/README.md" "${BUNDLE_DIR}/README_video_labeler.md"

echo "打源码备份"
tar -czf "${BUNDLE_DIR}/video_labeler_source.tar.gz" \
  -C "$(dirname "${SCRIPT_DIR}")" \
  --exclude='video_labeler/.git' \
  --exclude='video_labeler/__pycache__' \
  --exclude='video_labeler/logs' \
  --exclude='video_labeler/dist' \
  --exclude='video_labeler/*.log' \
  video_labeler

echo "生成校验文件"
(
  cd "${BUNDLE_DIR}"
  sha256sum \
    video-labeler_latest.tar \
    docker-compose.yml \
    label_groups.json \
    NAS_AGENT_HANDOFF.md \
    PROMPT_FOR_NAS_AGENT.md \
    README_video_labeler.md \
    video_labeler_source.tar.gz \
    > checksums.sha256
)

echo "生成总包"
tar -czf "${ARCHIVE_PATH}" -C "${DIST_DIR}" "${BUNDLE_NAME}"

echo
echo "部署目录: ${BUNDLE_DIR}"
echo "部署压缩包: ${ARCHIVE_PATH}"
