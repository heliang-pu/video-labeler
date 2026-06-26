"""CLI entrypoint for launching the browser-based Gradio app."""

from __future__ import annotations

import argparse

from .gradio_app import create_app


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the Gradio video labeler.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument(
        "--video-path",
        default="",
        help="Backward-compatible alias for the top video path or LeRobot dataset root.",
    )
    parser.add_argument(
        "--top-video-path",
        default="",
        help="Optional top-view video path, video directory, or LeRobot dataset root.",
    )
    parser.add_argument(
        "--wrist-video-path",
        default="",
        help="Optional wrist-view video path or directory.",
    )
    parser.add_argument(
        "--relative-video-path",
        default="",
        help="Optional relative path when the input is a video directory or LeRobot dataset root.",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link.")
    return parser


def main() -> None:
    """Launch the Gradio server."""
    args = build_parser().parse_args()
    default_top_video_path = args.top_video_path or args.video_path
    app = create_app(
        default_top_video_path=default_top_video_path,
        default_wrist_video_path=args.wrist_video_path,
        default_relative_path=args.relative_video_path,
    )
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        allowed_paths=["/workspace/dataset"],
    )


if __name__ == "__main__":
    main()
