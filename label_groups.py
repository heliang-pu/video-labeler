"""Persistent storage for reusable label groups."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import DEFAULT_LABEL_GROUP_NAME, DEFAULT_LABELS


LABEL_GROUP_STORE_VERSION = 1
DEFAULT_LABEL_GROUPS_PATH = Path(__file__).resolve().parent / "label_groups.json"


def normalize_label_list(labels: Iterable[str], *, require_non_empty: bool = True) -> list[str]:
    """Normalize labels by trimming whitespace and removing duplicates in order."""
    normalized_labels: list[str] = []
    seen_labels: set[str] = set()
    for label in labels:
        normalized_label = str(label).strip()
        if not normalized_label or normalized_label in seen_labels:
            continue
        normalized_labels.append(normalized_label)
        seen_labels.add(normalized_label)

    if require_non_empty and not normalized_labels:
        raise ValueError("标签组至少需要一个非空标签。")
    return normalized_labels


class LabelGroupRepository:
    """Load and persist named label groups for reuse across tasks."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = (
            DEFAULT_LABEL_GROUPS_PATH if storage_path is None else storage_path.expanduser().resolve()
        )

    def list_group_names(self) -> list[str]:
        """Return all available label-group names."""
        return list(self.load_groups().keys())

    def load_groups(self) -> dict[str, list[str]]:
        """Load stored label groups and merge them with the built-in default group."""
        groups = self._default_groups()
        if not self.storage_path.exists():
            return groups

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        raw_groups = payload.get("groups", payload)
        if not isinstance(raw_groups, dict):
            raise ValueError(f"标签组文件格式无效: {self.storage_path}")

        for raw_name, raw_labels in raw_groups.items():
            group_name = str(raw_name).strip()
            if not group_name or group_name == DEFAULT_LABEL_GROUP_NAME:
                continue
            try:
                normalized_labels = normalize_label_list(raw_labels)
            except (TypeError, ValueError):
                continue
            groups[group_name] = normalized_labels
        return groups

    def get_group_labels(self, group_name: str) -> list[str] | None:
        """Return one label group by name."""
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            return None
        labels = self.load_groups().get(normalized_group_name)
        return None if labels is None else list(labels)

    def save_group(self, group_name: str, labels: Iterable[str], *, overwrite: bool = False) -> tuple[str, bool]:
        """Persist one label group and report whether it overwrote an existing group."""
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ValueError("标签组名称不能为空。")
        if normalized_group_name == DEFAULT_LABEL_GROUP_NAME:
            raise ValueError(f"“{DEFAULT_LABEL_GROUP_NAME}”是内置标签组，请换一个名字。")

        normalized_labels = normalize_label_list(labels)
        groups = self.load_groups()
        already_exists = normalized_group_name in groups
        if already_exists and not overwrite:
            raise ValueError(f"标签组已存在: {normalized_group_name}")

        groups[normalized_group_name] = normalized_labels
        self._write_groups(groups)
        return normalized_group_name, already_exists

    def delete_group(self, group_name: str) -> None:
        """Delete one stored label group."""
        normalized_group_name = group_name.strip()
        if not normalized_group_name:
            raise ValueError("请先选择要删除的标签组。")
        if normalized_group_name == DEFAULT_LABEL_GROUP_NAME:
            raise ValueError(f"内置标签组“{DEFAULT_LABEL_GROUP_NAME}”不能删除。")

        groups = self.load_groups()
        if normalized_group_name not in groups:
            raise ValueError(f"标签组不存在: {normalized_group_name}")

        groups.pop(normalized_group_name)
        self._write_groups(groups)

    def _default_groups(self) -> dict[str, list[str]]:
        """Return built-in label groups that are always available."""
        return {DEFAULT_LABEL_GROUP_NAME: list(DEFAULT_LABELS)}

    def _write_groups(self, groups: dict[str, list[str]]) -> None:
        """Write only user-defined groups back to disk."""
        serializable_groups = {
            group_name: list(labels)
            for group_name, labels in groups.items()
            if group_name != DEFAULT_LABEL_GROUP_NAME
        }

        if not serializable_groups:
            if self.storage_path.exists():
                self.storage_path.unlink()
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LABEL_GROUP_STORE_VERSION,
            "groups": serializable_groups,
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
