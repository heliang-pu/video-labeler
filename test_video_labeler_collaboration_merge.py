"""Script-style smoke tests for merging per-episode sidecars."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_labeler.collaboration_merge import merge_episode_sublabels
from video_labeler.episode_metadata import EpisodeCatalog, EpisodeRange
from video_labeler.merge_episode_sublabels import main as merge_cli_main


def main() -> None:
    """Run merge-focused pure-Python checks."""
    with tempfile.TemporaryDirectory() as temp_dir_text:
        root = Path(temp_dir_text)
        catalog = EpisodeCatalog(
            dataset_root=root,
            fps=30.0,
            total_episodes=3,
            episodes={
                0: EpisodeRange(episode_index=0, length_frames=300, fps=30.0, duration_seconds=10.0),
                1: EpisodeRange(episode_index=1, length_frames=600, fps=30.0, duration_seconds=20.0),
                2: EpisodeRange(episode_index=2, length_frames=300, fps=30.0, duration_seconds=10.0),
            },
        )
        (root / "sublabels_episode_0000.json").write_text(
            json.dumps(
                {"sublabels": [{"start_seconds": 1.0, "end_seconds": 2.0, "label": "a"}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (root / "sublabels_episode_0002.json").write_text(
            json.dumps(
                {"sublabels": [{"start_seconds": 0.5, "end_seconds": 1.5, "label": "b"}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = merge_episode_sublabels(root, catalog)

        assert result.merged_episode_count == 2
        assert result.missing_episode_indices == [1]
        assert result.total_sublabel_count == 2
        assert result.label_counts == {"a": 1, "b": 1}
        payload = json.loads((root / "sublabels.json").read_text(encoding="utf-8"))
        assert payload["sublabels"] == [
            {"episode_index": 0, "start_seconds": 1.0, "end_seconds": 2.0, "label": "a"},
            {"episode_index": 2, "start_seconds": 30.5, "end_seconds": 31.5, "label": "b"},
        ]

        (root / "sublabels.json").unlink()
        with patch(
            "video_labeler.merge_episode_sublabels.load_episode_catalog",
            return_value=catalog,
        ), patch(
            "sys.argv",
            ["merge_episode_sublabels", str(root)],
        ):
            merge_cli_main()
        assert (root / "sublabels.json").exists()

    print("video_labeler collaboration merge smoke test passed.")


if __name__ == "__main__":
    main()
