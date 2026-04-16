"""Locate and validate COLMAP sparse reconstruction directories."""

from __future__ import annotations

from pathlib import Path


def colmap_sparse_candidate_dirs(data_dir: Path) -> list[Path]:
    """Ordered search roots (same order as training loader)."""
    return [
        data_dir / "sparse" / "0",
        data_dir / "undistorted" / "sparse",
        data_dir / "sparse",
        data_dir,
    ]


def find_colmap_sparse_dir(data_dir: Path) -> Path | None:
    """Return the first directory that contains a COLMAP camera model, or None."""
    for candidate in colmap_sparse_candidate_dirs(data_dir):
        if candidate.exists() and ((candidate / "cameras.txt").exists() or (candidate / "cameras.bin").exists()):
            return candidate
    return None


def require_colmap_sparse_model(data_dir: Path) -> Path:
    """Ensure a full sparse model (text or binary) exists; return its directory."""
    sparse_dir = find_colmap_sparse_dir(data_dir)
    if sparse_dir is None:
        raise FileNotFoundError(
            f"No COLMAP sparse model found under {data_dir}. "
            "Expected cameras.txt/bin under sparse/0/, undistorted/sparse/, or sparse/."
        )
    if (sparse_dir / "cameras.txt").exists():
        required = ("cameras.txt", "images.txt", "points3D.txt")
    else:
        required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required if not (sparse_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete COLMAP model in {sparse_dir}: missing {', '.join(missing)}")
    return sparse_dir
