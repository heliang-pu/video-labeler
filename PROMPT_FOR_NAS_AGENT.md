# Prompt For NAS Agent

把下面这段直接发给 NAS 上的 Claude/Codex：

```text
你现在在 NAS 上。请帮我部署一个 Docker 版机器人视频打标签工具。

部署包已经放在 NAS 的 video_labeler_deploy 目录里，包名类似：

video_labeler_nas_bundle_YYYYMMDD-HHMMSS

请按下面步骤做：

1. 进入最新的 video_labeler_nas_bundle_* 目录。
2. 先阅读 NAS_AGENT_HANDOFF.md。
3. 运行 sha256sum -c checksums.sha256 校验文件。
4. 用 docker load -i video-labeler_latest.tar 加载镜像。
5. 检查 NAS 上真实的数据集目录。这个共享目录里通常有 dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376。请把 docker-compose.yml 里的 volume 左侧路径改成 NAS 本机真实路径，右侧保持 /workspace/dataset。
6. 启动服务：docker compose up -d。如果系统只有 docker-compose，就用 docker-compose up -d。
7. 检查 docker ps、docker logs --tail 80 video-labeler、curl -I http://127.0.0.1:7860。
8. 最后告诉我浏览器访问地址，例如 http://<NAS_IP>:7860。

注意：
- 不要删除 dataset 里的 sublabels_episode_*.json 或 sublabels.json。
- 如果 7860 被占用，把 docker-compose.yml 的端口改成 7861:7860，并告诉我新地址。
- 如果 docker load 的镜像架构不兼容，请用 video_labeler_source.tar.gz 在 NAS 上重新 docker build。
- 多人标注时不需要在页面设置标注员，大家手动选择不同 episode 标注。
- 所有人标完后，用下面命令合并：

docker exec -it video-labeler python -m video_labeler.merge_episode_sublabels /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```
