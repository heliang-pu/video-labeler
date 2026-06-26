"""Browser-based Gradio frontend for the video time-range labeling workflow."""

from __future__ import annotations

import gradio as gr

from .gradio_session import DEFAULT_EXPORT_DIR, GradioLabelingSession, LEROBOT_DATASET_DIR
from .models import DEFAULT_LABEL_GROUP_NAME


APP_CSS = """
.vlm-sync-player {
  width: 100%;
}
.vlm-sync-videos {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.vlm-sync-panel {
  min-width: 0;
}
.vlm-sync-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 6px;
}
.vlm-sync-panel video {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: contain;
  background: #111;
  border-radius: 6px;
}
.vlm-sync-empty {
  padding: 16px;
  border: 1px solid #d0d7de;
  border-radius: 6px;
}
@media (max-width: 900px) {
  .vlm-sync-videos {
    grid-template-columns: 1fr;
  }
}
"""


def _time_slider_update(snapshot) -> dict:
    return gr.update(
        minimum=0,
        maximum=max(0.0, snapshot.total_seconds),
        value=min(snapshot.current_seconds, max(0.0, snapshot.total_seconds)),
        step=0.05,
        interactive=snapshot.frame_count > 0,
    )


def _dropdown_update(snapshot) -> dict:
    return gr.update(choices=snapshot.labels, value=snapshot.range_label)


def _label_group_update(snapshot) -> dict:
    return gr.update(choices=snapshot.label_group_names, value=snapshot.selected_label_group)


def _project_outputs(session: GradioLabelingSession):
    snapshot = session.snapshot()
    return (
        session,
        snapshot.sync_player_html,
        _time_slider_update(snapshot),
        gr.update(),
        gr.update(),
        snapshot.frame_info_markdown,
        snapshot.sublabels_table,
        _label_group_update(snapshot),
        _dropdown_update(snapshot),
        snapshot.label_summary_markdown,
        snapshot.project_markdown,
        snapshot.status_message,
    )


def _full_outputs(session: GradioLabelingSession):
    snapshot = session.snapshot()
    return (
        session,
        snapshot.top_input_path,
        snapshot.export_dir,
        snapshot.dataset_name,
        snapshot.instruction_template,
        snapshot.input_template,
        snapshot.frame_step,
        snapshot.keyframe_threshold,
        snapshot.sync_player_html,
        _time_slider_update(snapshot),
        snapshot.current_seconds,
        snapshot.current_seconds,
        snapshot.frame_info_markdown,
        snapshot.sublabels_table,
        _label_group_update(snapshot),
        snapshot.selected_label_group or "",
        _dropdown_update(snapshot),
        snapshot.label_summary_markdown,
        snapshot.project_markdown,
        snapshot.status_message,
    )


def _label_group_action_outputs(session: GradioLabelingSession):
    snapshot = session.snapshot()
    return (
        session,
        snapshot.selected_label_group or "",
        snapshot.sync_player_html,
        _time_slider_update(snapshot),
        gr.update(),
        gr.update(),
        snapshot.frame_info_markdown,
        snapshot.sublabels_table,
        _label_group_update(snapshot),
        _dropdown_update(snapshot),
        snapshot.label_summary_markdown,
        snapshot.project_markdown,
        snapshot.status_message,
    )


def create_app(
    default_top_video_path: str = "",
    default_wrist_video_path: str = "",
    default_relative_path: str = "",
) -> gr.Blocks:
    initial_session = GradioLabelingSession()
    initial_session.apply_label_group(DEFAULT_LABEL_GROUP_NAME)
    if default_top_video_path.strip():
        # Try episode-based loading first; fall back to direct load.
        loaded = False
        try:
            initial_session.load_episode(
                default_top_video_path,
                0,
                str(DEFAULT_EXPORT_DIR),
                initial_session.controller.project.dataset_name,
                initial_session.controller.project.instruction_template,
                initial_session.controller.project.input_template,
            )
            loaded = initial_session.controller.frame_count > 0
        except Exception:
            loaded = False
        if not loaded:
            initial_session.load_video(
                default_top_video_path,
                default_wrist_video_path,
                default_relative_path,
                str(DEFAULT_EXPORT_DIR),
                initial_session.controller.project.dataset_name,
                initial_session.controller.project.instruction_template,
                initial_session.controller.project.input_template,
            )
    initial_snapshot = initial_session.snapshot()
    initial_total_episodes = (
        initial_session.episode_catalog.total_episodes
        if initial_session.episode_catalog is not None
        else 0
    )

    with gr.Blocks(title="Video Labeler for LLaMA-Factory", css=APP_CSS) as demo:
        session_state = gr.State(initial_session)

        gr.Markdown(
            """
            # Video Labeler for LLaMA-Factory
            同步加载 `top`、`right_wrist`、`left_wrist` 三视角，用一条时间轴为视频片段添加秒级区间标签，并导出为 `LLaMA-Factory` 可用的 VLM 微调数据集。
            """
        )

        with gr.Row():
            top_video_path = gr.Textbox(
                label="LeRobot 数据集目录",
                value=initial_snapshot.top_input_path or default_top_video_path,
                placeholder=str(LEROBOT_DATASET_DIR / "fmc3_gr2_green_yellow_merged"),
                scale=6,
            )
            load_video_button = gr.Button("加载", variant="primary", scale=1)

        with gr.Row():
            export_dir = gr.Textbox(
                label="微调数据集导出目录",
                value=initial_snapshot.export_dir,
                placeholder=str(DEFAULT_EXPORT_DIR / "fmc3_gr2_green_yellow_merged"),
                scale=5,
            )
            export_button = gr.Button("导出微调数据集", variant="primary", scale=1)

        gr.Markdown("### 同步视频")
        sync_player = gr.HTML(value=initial_snapshot.sync_player_html, show_label=False)

        with gr.Row():
            prev_episode_button = gr.Button("◀ 上一个 Episode", scale=1)
            episode_number = gr.Number(
                label=(
                    f"当前 Episode (共 {initial_total_episodes} 个)"
                    if initial_total_episodes > 0
                    else "当前 Episode"
                ),
                value=initial_session.current_episode_index or 0,
                minimum=0,
                precision=0,
                scale=1,
            )
            next_episode_button = gr.Button("下一个 Episode ▶", scale=1)
            time_slider = gr.Slider(
                label="本 Episode 时间进度（秒）",
                minimum=0,
                maximum=max(0.0, initial_snapshot.total_seconds),
                value=min(initial_snapshot.current_seconds, max(0.0, initial_snapshot.total_seconds)),
                step=0.05,
                interactive=initial_snapshot.frame_count > 0,
                scale=6,
            )

        with gr.Row():
            jump_back_5s = gr.Button("⏪ -5s")
            jump_back_1s = gr.Button("◀ -1s")
            jump_forward_1s = gr.Button("▶ +1s")
            jump_forward_5s = gr.Button("⏩ +5s")

        with gr.Row():
            set_start_button = gr.Button("设为开始")
            set_end_button = gr.Button("设为结束")
            range_start_sec = gr.Number(
                label="开始（秒）",
                value=0.0,
                precision=3,
                scale=2,
            )
            range_end_sec = gr.Number(
                label="结束（秒）",
                value=0.0,
                precision=3,
                scale=2,
            )
            range_label_selector = gr.Dropdown(
                label="标签",
                choices=initial_snapshot.labels,
                value=initial_snapshot.range_label,
                scale=3,
            )
            apply_range_sec_button = gr.Button("添加区间标签", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=2):
                sublabels_table = gr.Dataframe(
                    label="已标注区间",
                    headers=["序号", "开始秒", "结束秒", "帧数量", "标签"],
                    datatype=["number", "number", "number", "number", "str"],
                    value=initial_snapshot.sublabels_table,
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    delete_row_index = gr.Number(
                        label="删除序号",
                        value=0,
                        minimum=0,
                        precision=0,
                        scale=1,
                    )
                    delete_sublabel_button = gr.Button("删除区间", scale=1)
                frame_info = gr.Markdown(initial_snapshot.frame_info_markdown)

            with gr.Column(scale=1):
                project_info = gr.Markdown(initial_snapshot.project_markdown)
                with gr.Row():
                    new_label = gr.Textbox(label="新增标签", placeholder="接近目标物", scale=4)
                    add_label_button = gr.Button("添加标签", scale=1)
                label_summary = gr.Markdown(initial_snapshot.label_summary_markdown)
                gr.Markdown("### 标签组")
                label_group_selector = gr.Dropdown(
                    label="已保存标签组",
                    choices=initial_snapshot.label_group_names,
                    value=initial_snapshot.selected_label_group,
                )
                with gr.Row():
                    apply_label_group_button = gr.Button("应用标签组")
                    delete_label_group_button = gr.Button("删除标签组")
                label_group_name = gr.Textbox(
                    label="标签组名称",
                    value=initial_snapshot.selected_label_group or "",
                    placeholder="例如: 抓取任务_状态标签",
                )
                with gr.Row():
                    save_label_group_button = gr.Button("保存为新组")
                    overwrite_label_group_button = gr.Button("覆盖保存")

        with gr.Accordion("高级设置", open=False):
            dataset_name = gr.Textbox(label="数据集名", value=initial_snapshot.dataset_name)
            frame_step = gr.Number(
                label="保留兼容步长",
                value=initial_snapshot.frame_step,
                minimum=1,
                precision=0,
            )
            keyframe_threshold = gr.Number(
                label="保留兼容关键帧阈值",
                value=initial_snapshot.keyframe_threshold,
                minimum=0,
                precision=2,
            )
            instruction_template = gr.Textbox(
                label="instruction",
                lines=3,
                value=initial_snapshot.instruction_template,
            )
            input_template = gr.Textbox(
                label="input 模板",
                lines=5,
                value=initial_snapshot.input_template,
            )

        status_box = gr.Markdown(initial_snapshot.status_message)

        _full_output_components = [
            session_state,
            top_video_path,
            export_dir,
            dataset_name,
            instruction_template,
            input_template,
            frame_step,
            keyframe_threshold,
            sync_player,
            time_slider,
            range_start_sec,
            range_end_sec,
            frame_info,
            sublabels_table,
            label_group_selector,
            label_group_name,
            range_label_selector,
            label_summary,
            project_info,
            status_box,
        ]

        _project_output_components = [
            session_state,
            sync_player,
            time_slider,
            range_start_sec,
            range_end_sec,
            frame_info,
            sublabels_table,
            label_group_selector,
            range_label_selector,
            label_summary,
            project_info,
            status_box,
        ]

        _label_group_action_output_components = [
            session_state,
            label_group_name,
            sync_player,
            time_slider,
            range_start_sec,
            range_end_sec,
            frame_info,
            sublabels_table,
            label_group_selector,
            range_label_selector,
            label_summary,
            project_info,
            status_box,
        ]

        sync_inputs = [session_state, dataset_name, instruction_template, input_template]

        def sync_templates(session, dataset_text, instruction_text, input_text):
            session.sync_export_settings(dataset_text, instruction_text, input_text)
            return session

        for component in (dataset_name, instruction_template, input_template):
            component.change(sync_templates, inputs=sync_inputs, outputs=[session_state], queue=False)

        def handle_frame_step_change(session, v):
            session.set_frame_step(int(v))
            snapshot = session.snapshot()
            return session, snapshot.frame_info_markdown, snapshot.project_markdown, snapshot.status_message

        frame_step.change(
            handle_frame_step_change,
            inputs=[session_state, frame_step],
            outputs=[session_state, frame_info, project_info, status_box],
            queue=False,
        )

        def handle_keyframe_threshold_change(session, v):
            session.set_keyframe_threshold(float(v))
            snapshot = session.snapshot()
            return session, snapshot.frame_info_markdown, snapshot.project_markdown, snapshot.status_message

        keyframe_threshold.change(
            handle_keyframe_threshold_change,
            inputs=[session_state, keyframe_threshold],
            outputs=[session_state, frame_info, project_info, status_box],
            queue=False,
        )

        def handle_load_video(session, top_text, episode_num, export_text, dataset_text, instruction_text, input_text):
            session.load_episode(top_text, int(episode_num), export_text, dataset_text, instruction_text, input_text)
            outputs = list(_full_outputs(session))
            return (*outputs, session.current_episode_index or 0)

        def handle_initial_load(session, top_text, episode_num, export_text, dataset_text, instruction_text, input_text):
            if not str(top_text).strip():
                outputs = list(_full_outputs(session))
                return (*outputs, session.current_episode_index or 0)
            session.load_episode(top_text, int(episode_num), export_text, dataset_text, instruction_text, input_text)
            outputs = list(_full_outputs(session))
            return (*outputs, session.current_episode_index or 0)

        load_video_button.click(
            handle_load_video,
            inputs=[session_state, top_video_path, episode_number, export_dir, dataset_name, instruction_template, input_template],
            outputs=_full_output_components + [episode_number],
            queue=False,
        )

        demo.load(
            handle_initial_load,
            inputs=[session_state, top_video_path, episode_number, export_dir, dataset_name, instruction_template, input_template],
            outputs=_full_output_components + [episode_number],
            queue=False,
        )

        def handle_prev_episode(session, top_text, episode_num, export_text, dataset_text, instruction_text, input_text):
            session.current_episode_index = int(episode_num)
            target_episode = max(0, int(episode_num) - 1)
            session.load_episode(top_text, target_episode, export_text, dataset_text, instruction_text, input_text)
            outputs = list(_full_outputs(session))
            new_episode = session.current_episode_index or 0
            return (*outputs, new_episode)

        def handle_next_episode(session, top_text, episode_num, export_text, dataset_text, instruction_text, input_text):
            session.current_episode_index = int(episode_num)
            target_episode = int(episode_num) + 1
            if session.episode_catalog is not None:
                target_episode = min(target_episode, session.episode_catalog.total_episodes - 1)
            session.load_episode(top_text, target_episode, export_text, dataset_text, instruction_text, input_text)
            outputs = list(_full_outputs(session))
            new_episode = session.current_episode_index or 0
            return (*outputs, new_episode)

        prev_episode_button.click(
            handle_prev_episode,
            inputs=[session_state, top_video_path, episode_number, export_dir, dataset_name, instruction_template, input_template],
            outputs=_full_output_components + [episode_number],
            queue=False,
        )

        next_episode_button.click(
            handle_next_episode,
            inputs=[session_state, top_video_path, episode_number, export_dir, dataset_name, instruction_template, input_template],
            outputs=_full_output_components + [episode_number],
            queue=False,
        )

        def handle_export(session, export_text, dataset_text, instruction_text, input_text, progress=gr.Progress()):
            progress(0, desc="准备导出...")
            def progress_callback(current: int, total: int, desc: str = ""):
                if total > 0:
                    progress(current / total, desc=desc)
                else:
                    progress(0, desc=desc)
            session.export_dataset(export_text, dataset_text, instruction_text, input_text, progress_callback=progress_callback)
            progress(1.0, desc="导出完成")
            return _full_outputs(session)

        export_button.click(
            handle_export,
            inputs=[session_state, export_dir, dataset_name, instruction_template, input_template],
            outputs=_full_output_components,
            queue=True,
        )

        def handle_jump_seconds(session, seconds):
            session.jump_to_seconds(float(seconds))
            return _project_outputs(session)

        time_slider.release(
            handle_jump_seconds,
            inputs=[session_state, time_slider],
            outputs=_project_output_components,
            queue=False,
        )

        def handle_jump_back_5s(session):
            session.jump_relative_seconds(-5.0)
            return _project_outputs(session)

        def handle_jump_back_1s(session):
            session.jump_relative_seconds(-1.0)
            return _project_outputs(session)

        def handle_jump_forward_1s(session):
            session.jump_relative_seconds(1.0)
            return _project_outputs(session)

        def handle_jump_forward_5s(session):
            session.jump_relative_seconds(5.0)
            return _project_outputs(session)

        jump_back_5s.click(handle_jump_back_5s, inputs=[session_state],
                          outputs=_project_output_components, queue=False)
        jump_back_1s.click(handle_jump_back_1s, inputs=[session_state],
                          outputs=_project_output_components, queue=False)
        jump_forward_1s.click(handle_jump_forward_1s, inputs=[session_state],
                             outputs=_project_output_components, queue=False)
        jump_forward_5s.click(handle_jump_forward_5s, inputs=[session_state],
                             outputs=_project_output_components, queue=False)

        def handle_set_start(session):
            return session.current_seconds()

        def handle_set_end(session):
            return session.current_seconds()

        set_start_button.click(handle_set_start, inputs=[session_state], outputs=[range_start_sec], queue=False)
        set_end_button.click(handle_set_end, inputs=[session_state], outputs=[range_end_sec], queue=False)

        def handle_apply_range_seconds(session, start_sec, end_sec, label_value):
            session.apply_range_label_seconds(float(start_sec), float(end_sec), label_value)
            return _project_outputs(session)

        apply_range_sec_button.click(
            handle_apply_range_seconds,
            inputs=[session_state, range_start_sec, range_end_sec, range_label_selector],
            outputs=_project_output_components,
            queue=False,
        )

        def handle_delete_sublabel(session, row_index):
            session.delete_sublabel(int(row_index))
            return _project_outputs(session)

        delete_sublabel_button.click(
            handle_delete_sublabel,
            inputs=[session_state, delete_row_index],
            outputs=_project_output_components,
            queue=False,
        )

        def handle_range_label_change(session, label_value):
            session.set_range_label(label_value)
            return session

        range_label_selector.change(
            handle_range_label_change,
            inputs=[session_state, range_label_selector],
            outputs=[session_state],
            queue=False,
        )

        def handle_add_label(session, label_text):
            session.add_label(label_text)
            snapshot = session.snapshot()
            return (
                session,
                gr.update(value=""),
                _label_group_update(snapshot),
                _dropdown_update(snapshot),
                snapshot.label_summary_markdown,
                snapshot.project_markdown,
                snapshot.status_message,
            )

        add_label_button.click(
            handle_add_label,
            inputs=[session_state, new_label],
            outputs=[
                session_state,
                new_label,
                label_group_selector,
                range_label_selector,
                label_summary,
                project_info,
                status_box,
            ],
            queue=False,
        )

        def handle_label_group_change(session, group_name):
            session.set_selected_label_group(group_name)
            snapshot = session.snapshot()
            return session, gr.update(value=snapshot.selected_label_group or "")

        label_group_selector.change(
            handle_label_group_change,
            inputs=[session_state, label_group_selector],
            outputs=[session_state, label_group_name],
            queue=False,
        )

        def handle_apply_label_group(session, group_name):
            session.apply_label_group(group_name)
            return _label_group_action_outputs(session)

        apply_label_group_button.click(
            handle_apply_label_group,
            inputs=[session_state, label_group_selector],
            outputs=_label_group_action_output_components,
            queue=False,
        )

        def handle_save_label_group(session, group_name_text):
            session.save_current_labels_as_group(group_name_text, overwrite=False)
            return _label_group_action_outputs(session)

        save_label_group_button.click(
            handle_save_label_group,
            inputs=[session_state, label_group_name],
            outputs=_label_group_action_output_components,
            queue=False,
        )

        def handle_overwrite_label_group(session, group_name_text):
            session.save_current_labels_as_group(group_name_text, overwrite=True)
            return _label_group_action_outputs(session)

        overwrite_label_group_button.click(
            handle_overwrite_label_group,
            inputs=[session_state, label_group_name],
            outputs=_label_group_action_output_components,
            queue=False,
        )

        def handle_delete_label_group(session, group_name):
            session.delete_label_group(group_name)
            return _label_group_action_outputs(session)

        delete_label_group_button.click(
            handle_delete_label_group,
            inputs=[session_state, label_group_selector],
            outputs=_label_group_action_output_components,
            queue=False,
        )

    return demo
