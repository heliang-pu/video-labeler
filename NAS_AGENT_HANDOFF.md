# NAS Agent Handoff: Video Labeler Docker Deployment

这份文档是给 NAS 上的 Claude/Codex/运维代理看的。目标是在 NAS 上部署 Gradio 版多视角视频区间标注器，让多人通过浏览器标注 LeRobot episode 数据集。

## 你拿到的文件

部署包通常包含：

```text
video-labeler_latest.tar      # 已构建好的 Docker 镜像
docker-compose.yml            # NAS 部署 compose，来自 docker-compose.nas.yml
label_groups.json             # 默认标签组配置
NAS_AGENT_HANDOFF.md          # 本文件
README_video_labeler.md       # 完整项目说明
video_labeler_source.tar.gz   # 源码备份，镜像不兼容时可重新构建
checksums.sha256              # 文件校验
logs/                         # 容器日志挂载目录
```

## 重要默认值

容器内数据集路径固定建议保持：

```text
/workspace/dataset
```

默认加载的数据集是：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

所以 NAS 宿主机的数据目录里应该有：

```text
<NAS数据目录>/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

默认服务端口：

```text
7860
```

## 部署步骤

进入部署包目录：

```bash
cd /path/to/video_labeler_nas_bundle
```

先按需校验文件：

```bash
sha256sum -c checksums.sha256
```

加载镜像：

```bash
docker load -i video-labeler_latest.tar
```

编辑 `docker-compose.yml`，把左边 NAS 路径换成真实数据集目录：

```yaml
volumes:
  - /volume1/robot_datasets:/workspace/dataset
```

例如数据在 `/volume1/dataset`，就改成：

```yaml
volumes:
  - /volume1/dataset:/workspace/dataset
```

启动：

```bash
mkdir -p logs
docker compose up -d
```

如果 NAS 上是老版本 Docker Compose：

```bash
docker-compose up -d
```

检查容器：

```bash
docker ps --filter name=video-labeler
docker logs --tail 80 video-labeler
```

检查 HTTP：

```bash
curl -I http://127.0.0.1:7860
```

返回 `200` 或 `302` 都说明服务起来了。

浏览器访问：

```text
http://<NAS_IP>:7860
```

## 多人标注约定

页面里不需要设置标注员。直接人工分 episode：

```text
人员 A: episode 0-49
人员 B: episode 50-99
人员 C: episode 100-149
```

每个人打开同一个 URL，手动选择自己的 episode 标。不要两个人同时标同一个 episode。

每个 episode 会自动保存：

```text
<数据集目录>/sublabels_episode_0000.json
<数据集目录>/sublabels_episode_0001.json
...
```

## 合并标注

所有人标完后在 NAS 上执行：

```bash
docker exec -it video-labeler python -m video_labeler.merge_episode_sublabels \
  /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

合并结果会写到：

```text
<数据集目录>/sublabels.json
```

合并逻辑会把每个 episode 内的局部秒数转换为整个数据集的全局秒数。

## 标签任务

默认标签组：

```text
parcel_sorting_bimanual
```

默认标签：

```text
left_hand_pickup
object_at_center
right_hand_transfer
```

任务目标：根据 `top`、`right_wrist`、`left_wrist` 三视角判断当前机器人任务阶段，只输出标签名。

## 常见问题

端口被占用：

把 compose 里的端口从：

```yaml
ports:
  - "7860:7860"
```

改成：

```yaml
ports:
  - "7861:7860"
```

然后访问 `http://<NAS_IP>:7861`。

页面提示找不到数据：

检查 `docker-compose.yml` 的 volume 左边路径是否真实存在，右边是否仍是 `/workspace/dataset`。

容器权限导致无法写 `sublabels_episode_XXXX.json`：

给 NAS 数据目录开放容器写入权限，或者用 NAS 管理界面给 Docker 用户写权限。

镜像架构不兼容：

当前镜像通常是在 x86_64 Linux 上构建。如果 NAS 是 ARM/aarch64，可能需要用源码重新构建：

```bash
tar -xzf video_labeler_source.tar.gz
cd video_labeler
docker buildx build --platform linux/arm64 -t video-labeler:latest --load .
```

如果 NAS 是 x86_64，直接 `docker load` 通常即可。

## 重新部署或升级

停止旧容器：

```bash
docker compose down
```

重新加载新镜像：

```bash
docker load -i video-labeler_latest.tar
```

启动：

```bash
docker compose up -d
```

不要删除数据集目录里的 `sublabels_episode_*.json`、`sublabels.json`，这些是标注成果。
