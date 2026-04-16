"""Tests for COLMAP sparse directory discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.preprocess.colmap_ready import (
    find_colmap_sparse_dir,
    require_colmap_sparse_model,
)


def test_find_prefers_sparse_zero(tmp_path: Path):
    """Should pick sparse/0 when it contains cameras.txt."""
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    (tmp_path / "sparse" / "0" / "cameras.txt").write_text("#\n")
    assert find_colmap_sparse_dir(tmp_path) == tmp_path / "sparse" / "0"


def test_require_text_triplet(tmp_path: Path):
    """Text model needs cameras, images, points3D."""
    sd = tmp_path / "sparse" / "0"
    sd.mkdir(parents=True)
    (sd / "cameras.txt").write_text("#\n")
    (sd / "images.txt").write_text("#\n")
    (sd / "points3D.txt").write_text("#\n")
    assert require_colmap_sparse_model(tmp_path) == sd


def test_require_raises_if_incomplete(tmp_path: Path):
    sd = tmp_path / "sparse" / "0"
    sd.mkdir(parents=True)
    (sd / "cameras.txt").write_text("#\n")
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        require_colmap_sparse_model(tmp_path)
