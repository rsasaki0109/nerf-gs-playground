"""Helpers for the GitHub Pages production scene manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "gs-mapper-scene-picker/v1"


@dataclass(frozen=True)
class PageScene:
    """One production splat exposed by the public Pages viewers."""

    url: str
    preview: str
    label: str
    summary: str

    @property
    def preview_stem(self) -> str:
        return Path(self.preview).stem


def _require_relative_path(value: str, *, field: str, index: int) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"scene {index} has unsafe {field}: {value}")
    return value


def _require_string(raw: dict[str, Any], field: str, *, index: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"scene {index} is missing string field {field!r}")
    return value


def load_page_scenes(docs_dir: Path) -> list[PageScene]:
    """Load production scenes from docs/scenes-list.json."""
    manifest_path = docs_dir / "scenes-list.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported scene manifest version: {data.get('version')!r}")
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("scene manifest must contain at least one scene")

    scenes: list[PageScene] = []
    for index, raw in enumerate(raw_scenes, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"scene {index} must be an object")
        url = _require_relative_path(_require_string(raw, "url", index=index), field="url", index=index)
        preview = _require_relative_path(_require_string(raw, "preview", index=index), field="preview", index=index)
        scenes.append(
            PageScene(
                url=url,
                preview=preview,
                label=_require_string(raw, "label", index=index),
                summary=_require_string(raw, "summary", index=index),
            )
        )
    return scenes


def scene_urls(docs_dir: Path) -> list[str]:
    """Return production scene URLs in picker/hero order."""
    return [scene.url for scene in load_page_scenes(docs_dir)]


def capture_scene_specs(docs_dir: Path) -> list[tuple[str, str]]:
    """Return (preview filename stem, splat URL) pairs for README captures."""
    return [(scene.preview_stem, scene.url) for scene in load_page_scenes(docs_dir)]
