# Video Labeler

浏览器版多视角视频区间标注工具，项目根目录：

```text
/workspace/video_labeler
```

它现在主要服务这条闭环：

```text
LeRobot episode 数据集
  -> 浏览器按秒标注阶段区间
  -> 自动保存 sublabels_episode_XXXX.json
  -> 合并成 sublabels.json
  -> 导出 Qwen2.5-VL / LLaMA-Factory 微调数据
  -> 训练 LoRA
  -> 推理测试阶段标签
```

当前正在用的数据集示例：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

当前远程访问链接：

```text
http://<你的机器IP>:7860
```

## 当前任务目标

当前标签任务是让 VLM 根据三视角画面判断机器人任务阶段，只输出阶段标签名。

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

默认 instruction：

```text
Based on three camera views (top, right_wrist, left_wrist), determine the current phase of the robot task.

Task: Parcel Sorting (Bimanual Coordination)
- left_hand_pickup: Left hand picks an object from the pile and places it at the center area
- object_at_center: Object is placed at center area, ready for right hand pickup
- right_hand_transfer: Right hand picks from center and transfers to the right side

Output only the phase label name.
```

默认 input 模板（三视角）：

```text
top视角: <image>
right_wrist视角: <image>
left_wrist视角: <image>
```

## 环境准备

第一次使用：

```bash
bash /workspace/video_labeler/setup_video_labeler.sh
```

后续使用推荐：

```bash
source /workspace/video_labeler/shell_utils.sh
activate_video_labeler_env
```

脚本会优先使用 `video_labeler` conda 环境；如果本机已经在用 `qwen25vl_factory_qt` 或 `qwen25vl_factory`，也会兼容。

## 启动标注器

推荐命令：

```bash
source /workspace/video_labeler/shell_utils.sh
activate_video_labeler_env

PYTHONPATH=/workspace python -m video_labeler.main_gradio \
  --host 0.0.0.0 \
  --port 7860 \
  --top-video-path /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

或者用封装脚本：

```bash
bash /workspace/video_labeler/run_video_labeler.sh \
  /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

页面默认监听：

```text
0.0.0.0:7860
```

Tailscale 远程打开：

```text
http://<你的机器IP>:7860
```

本机或端口转发打开：

```text
http://127.0.0.1:7860
```

后台启动示例：

```bash
mkdir -p /workspace/video_labeler/logs

setsid bash -lc 'source /workspace/video_labeler/shell_utils.sh && activate_video_labeler_env && cd /workspace && PYTHONPATH=/workspace python -m video_labeler.main_gradio --host 0.0.0.0 --port 7860 --top-video-path /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376' \
  > /workspace/video_labeler/logs/gradio_7860.log 2>&1 < /dev/null &
```

确认服务是否活着：

```bash
python - <<'PY'
from urllib.request import urlopen
resp = urlopen("http://127.0.0.1:7860", timeout=5)
print(resp.status)
PY
```

返回 `200` 就是正常。

## 页面怎么标

1. 打开页面。
2. `LeRobot 数据集目录` 填数据集根目录，例如：

   ```text
   /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
   ```

3. 选择 `当前 Episode`，点击加载。
4. 用时间轴或 `-5s / -1s / +1s / +5s` 找到片段开始。
5. 点击 `设为开始`。
6. 找到片段结束。
7. 点击 `设为结束`。
8. 选择标签，点击 `添加区间标签`。
9. 表格里会出现一条区间，第四列是该秒区间覆盖的帧数量。

现在是秒区间标注，不是逐帧点标签。秒可以带小数，例如：

```text
0.233
1.467
```

这是正常的，方便按视频帧率精确对齐。

## 自动保存

只要当前加载的是 LeRobot episode 数据集，添加或删除区间后会自动保存。

每个 episode 单独保存：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/sublabels_episode_0000.json
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/sublabels_episode_0001.json
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/sublabels_episode_0002.json
...
```

这样多人标注时互相不覆盖。

如果加载整个拼接视频或旧目录模式，也会保存到：

```text
sublabels.json
```

但当前推荐用 episode 模式。

## 多人标注流程

现在页面里不再做“标注员分配”。最简单可靠的流程是人工约定 episode 范围：

```text
人员 A: episode 0-49
人员 B: episode 50-99
人员 C: episode 100-149
```

每个人打开同一个链接，手动输入自己的 episode 编号标注即可。

注意：

- 不要两个人同时标同一个 episode。
- 每个人标完一个 episode 后，可以点下一个 episode 继续。
- 每个 episode 会自动写自己的 `sublabels_episode_XXXX.json`。

## 合并多人标注

所有人标完后，用命令行合并。

```bash
source /workspace/video_labeler/shell_utils.sh
activate_video_labeler_env

PYTHONPATH=/workspace python -m video_labeler.merge_episode_sublabels \
  /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

输出会类似：

```text
输出: /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/sublabels.json
已合并 episode: 2/3
区间标签数: 2
标签统计: left_hand_pickup:1, object_at_center:1
未标/空标 episode: 1
```

合并结果：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/sublabels.json
```

合并逻辑：

- 读取所有 `sublabels_episode_XXXX.json`
- 按 episode 顺序拼成全局时间线
- episode 内的本地秒数会转换成全数据集全局秒数
- 空 episode 或没标的 episode 会列到 `missing_episode_indices`

## 导出微调数据集

合并后回到页面，点击：

```text
导出微调数据集
```

默认导出目录：

```text
/workspace/dataset/qwen_finetune/camera_pen_touch_clean_del_52_376
```

导出内容包括：

```text
dataset.json
dataset_info.json
train.json
val.json
test.json
annotations.json
sublabels.json
workspace_meta.json
export_report.json
images/
train_qwen25vl_lora.yaml
training_readme.md
```

关键点：

- 秒区间会导出区间覆盖的所有帧，不再是每秒抽一张。
- 每条样本包含三张图：`top`、`right_wrist`、`left_wrist`。
- 输出标签就是区间标签名。

单条样本格式类似：

```json
{
  "instruction": "Based on three camera views ...",
  "input": "top视角: <image>\nright_wrist视角: <image>\nleft_wrist视角: <image>",
  "output": "left_hand_pickup",
  "images": [
    "images/top/range_0000_sample_000_file-000_seg000_frame000010.jpg",
    "images/wrist/range_0000_sample_000_file-000_seg000_frame000010.jpg",
    "images/left_wrist/range_0000_sample_000_file-000_seg000_frame000010.jpg"
  ]
}
```

## 训练 LoRA

导出完成后，直接使用导出目录里的 YAML 训练：

```bash
source /workspace/video_finetune/shell_utils.sh
activate_video_finetune_env

bash /workspace/video_finetune/run_qwen25vl_lora_train.sh \
  /workspace/dataset/qwen_finetune/camera_pen_touch_clean_del_52_376/train_qwen25vl_lora.yaml
```

训练输出默认在：

```text
/workspace/video_finetune/outputs/camera_pen_touch_clean_del_52_376_qwen25vl_lora
```

导出目录里的 `training_readme.md` 也会写好训练命令。

## 推理测试

训练后可以测试某个时间点：

```bash
source /workspace/video_finetune/shell_utils.sh
activate_video_finetune_env

python /workspace/video_finetune/test_phase_inference.py \
  --lora_path /workspace/video_finetune/outputs/camera_pen_touch_clean_del_52_376_qwen25vl_lora \
  --video_dir /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/videos \
  --timestamp 0
```

推荐使用当前脚本的 score 模式，它会在合法标签里打分选择：

```text
left_hand_pickup
object_at_center
right_hand_transfer
```

示例输出：

```text
判断结果: left_hand_pickup
推理方式: score
标签分数: left_hand_pickup=-0.124, object_at_center=-0.332, right_hand_transfer=-1.333
```

## 常用路径

当前标注源数据：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

当前微调导出数据：

```text
/workspace/dataset/qwen_finetune/camera_pen_touch_clean_del_52_376
```

当前 LoRA 输出：

```text
/workspace/video_finetune/outputs/camera_pen_touch_clean_del_52_376_qwen25vl_lora
```

标注器日志：

```text
/workspace/video_labeler/logs/gradio_7860.log
```

标签组：

```text
/workspace/video_labeler/label_groups.json
```

## Docker 部署到 NAS

容器只运行标注器服务，数据集不打进镜像，通过 volume 挂载到容器内：

```text
/workspace/dataset
```

### 目录文件

Docker 相关文件在：

```text
/workspace/video_labeler/Dockerfile
/workspace/video_labeler/docker-compose.yml
/workspace/video_labeler/docker-compose.nas.yml
/workspace/video_labeler/.dockerignore
```

### 在当前机器构建镜像

```bash
cd /workspace/video_labeler
docker build -t video-labeler:latest .
```

本机直接用 compose 启动：

```bash
cd /workspace/video_labeler
docker compose up -d
```

本机 compose 默认挂载：

```text
/workspace/dataset:/workspace/dataset
```

访问：

```text
http://<机器IP>:7860
```

查看日志：

```bash
docker logs -f video-labeler
```

停止：

```bash
docker compose down
```

### 传到 NAS

如果 NAS 能直接访问这个代码目录，可以在 NAS 上执行同样的 `docker build`。

如果需要打包传镜像：

```bash
cd /workspace/video_labeler
docker build -t video-labeler:latest .
docker save video-labeler:latest -o video-labeler_latest.tar
```

把 `video-labeler_latest.tar` 上传到 NAS 后：

```bash
docker load -i video-labeler_latest.tar
```

### NAS compose 配置

复制 compose 示例：

```bash
cp /workspace/video_labeler/docker-compose.nas.yml /path/on/nas/docker-compose.yml
```

编辑 `docker-compose.yml`，把左边 NAS 路径改成你的真实数据集目录：

```yaml
volumes:
  - /volume1/robot_datasets:/workspace/dataset
```

容器内路径建议保持：

```text
/workspace/dataset
```

这样 README 里的默认命令和页面默认路径都不用改。

如果你的 NAS 数据实际放在：

```text
/volume1/dataset
```

就写：

```yaml
volumes:
  - /volume1/dataset:/workspace/dataset
```

启动：

```bash
docker compose up -d
```

访问：

```text
http://<NAS_IP>:7860
```

### NAS 上的数据目录要求

容器默认加载：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

所以 NAS 的宿主机目录里应有：

```text
<NAS数据目录>/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

例如：

```text
/volume1/robot_datasets/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

容器看到的就是：

```text
/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

### 在容器里合并 episode 标注

多人标完后，在 NAS 上执行：

```bash
docker exec -it video-labeler \
  python -m video_labeler.merge_episode_sublabels \
  /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

它会在挂载的数据集目录里生成：

```text
sublabels.json
```

### 在容器里验证服务

```bash
docker exec -it video-labeler python - <<'PY'
from urllib.request import urlopen
resp = urlopen("http://127.0.0.1:7860", timeout=5)
print(resp.status)
PY
```

返回 `200` 表示服务正常。

### NAS 注意事项

- NAS 如果是 x86_64，直接用上面的 `python:3.11-slim` 镜像即可。
- NAS 如果是 ARM，需要用 `docker buildx` 构建对应架构镜像。
- 数据目录必须可写，因为标注会写 `sublabels_episode_XXXX.json`。
- 不要把数据集 COPY 进镜像；一定用 volume。
- 如果端口 `7860` 被占用，把 compose 里的 `"7860:7860"` 改成例如 `"7861:7860"`。

## 常用命令

查看服务：

```bash
ps -ef | rg 'video_labeler.main_gradio|7860'
```

停止服务：

```bash
kill <PID>
```

重启服务：

```bash
setsid bash -lc 'source /workspace/video_labeler/shell_utils.sh && activate_video_labeler_env && cd /workspace && PYTHONPATH=/workspace python -m video_labeler.main_gradio --host 0.0.0.0 --port 7860 --top-video-path /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376' \
  > /workspace/video_labeler/logs/gradio_7860.log 2>&1 < /dev/null &
```

合并 episode 标注：

```bash
PYTHONPATH=/workspace python -m video_labeler.merge_episode_sublabels \
  /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376
```

训练：

```bash
bash /workspace/video_finetune/run_qwen25vl_lora_train.sh \
  /workspace/dataset/qwen_finetune/camera_pen_touch_clean_del_52_376/train_qwen25vl_lora.yaml
```

推理：

```bash
python /workspace/video_finetune/test_phase_inference.py \
  --lora_path /workspace/video_finetune/outputs/camera_pen_touch_clean_del_52_376_qwen25vl_lora \
  --video_dir /workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376/videos \
  --timestamp 0
```

## 测试

改代码后建议跑：

```bash
source /workspace/video_labeler/shell_utils.sh
activate_video_labeler_env

PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_workspace.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_export.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_collaboration_merge.py
PYTHONPATH=/workspace python /workspace/video_labeler/test_video_labeler_time_ranges.py
```

快速检查 Gradio app 能否构建：

```bash
PYTHONPATH=/workspace python - <<'PY'
from video_labeler.gradio_app import create_app
app = create_app()
print(len(app.fns))
PY
```

## 注意事项

- 两个人不要同时标同一个 episode。
- 标完后先合并，再导出训练集。
- 导出训练集会产生很多帧图片，区间越长，样本越多。
- 如果页面还是旧 UI，强刷浏览器缓存。
- matplotlib 中文字体 warning 不影响标注和保存。
