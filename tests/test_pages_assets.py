"""Sanity checks for GitHub Pages static assets (docs/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assets_dir() -> Path:
    return REPO_ROOT / "docs" / "assets"


def test_scenes_index_json_parses(assets_dir: Path) -> None:
    p = assets_dir / "scenes.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("type") == "web-scene-index"
    assert isinstance(data.get("scenes"), list)


def test_scenes_index_lists_outdoor_demo_first(assets_dir: Path) -> None:
    data = json.loads((assets_dir / "scenes.json").read_text(encoding="utf-8"))
    scenes = data["scenes"]
    assert len(scenes) >= 1
    assert scenes[0]["id"] == "outdoor-demo"
    assert scenes[0]["manifest"] == "assets/outdoor-demo/scene.json"


def test_outdoor_demo_manifest_and_referenced_binary_exist(assets_dir: Path) -> None:
    manifest_path = assets_dir / "outdoor-demo" / "scene.json"
    assert manifest_path.is_file()
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert man.get("type") == "web-scene-manifest"
    href = man["asset"]["href"]
    bin_path = (manifest_path.parent / href).resolve()
    assert bin_path.is_file(), f"missing binary referenced by outdoor-demo: {href}"


def test_index_html_links_outdoor_demo_scene() -> None:
    """Landing page keeps deep links to the Outdoor GS GitHub Pages tab."""
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "?scene=outdoor-demo" in html
    assert "assets/scenes.json" in html


def test_docs_scenes_metadata_includes_outdoor() -> None:
    """Gallery metadata JSON documents the outdoor demo entry."""
    data = json.loads((REPO_ROOT / "docs" / "scenes.json").read_text(encoding="utf-8"))
    assert "outdoor" in data
    assert "Outdoor" in data["outdoor"]["name"] or "outdoor" in data["outdoor"]["name"].lower()


def test_readme_avoids_local_machine_paths() -> None:
    """README links should be portable (no /media/... absolute paths)."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "/media/" not in text
