"""Import artifacts produced by external visual SLAM front-ends.

This module intentionally does not import MASt3R-SLAM, VGGT-SLAM, LoGeR,
Pi3, or their model dependencies. It only normalizes their exported trajectory
and optional point-cloud files into the existing COLMAP text import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_DEFAULT_TRAJECTORY_CANDIDATES = (
    "poses.txt",
    "trajectory.txt",
    "traj.txt",
    "output_txt.txt",
    "*.tum",
    "*.traj",
    "*.txt",
)

_DEFAULT_POINTCLOUD_CANDIDATES = (
    "points.ply",
    "pointcloud.ply",
    "map.ply",
    "reconstruction.ply",
    "*_points.pcd",
    "*.ply",
    "*.npy",
    "*.pcd",
)

_SKIP_TEXT_NAMES = ("readme", "license", "config", "metrics", "results", "log")
_POSE_TENSOR_SUFFIXES = (".npy", ".npz", ".pt", ".pth")
_POSE_MATRIX_KEYS = (
    "camera_poses",
    "camera_pose",
    "poses",
    "pose",
    "Twc",
    "T_wc",
    "c2w",
    "cam2world",
    "camera_to_world",
)
_TIMESTAMP_KEYS = ("timestamps", "timestamp", "times", "frame_ids", "indices")


@dataclass(frozen=True, slots=True)
class ExternalSLAMProfile:
    """Known artifact conventions for an external SLAM/reconstruction system."""

    key: str
    display_name: str
    default_trajectory_format: str
    trajectory_candidates: tuple[str, ...]
    pointcloud_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExternalSLAMArtifacts:
    """Resolved files ready for conversion through the existing trajectory importer."""

    system: str
    trajectory_path: Path
    trajectory_format: str
    pointcloud_path: Path | None = None
    pinhole_calib_path: Path | None = None


PROFILES: dict[str, ExternalSLAMProfile] = {
    "generic": ExternalSLAMProfile(
        key="generic",
        display_name="external SLAM",
        default_trajectory_format="tum",
        trajectory_candidates=_DEFAULT_TRAJECTORY_CANDIDATES,
        pointcloud_candidates=_DEFAULT_POINTCLOUD_CANDIDATES,
    ),
    "mast3r-slam": ExternalSLAMProfile(
        key="mast3r-slam",
        display_name="MASt3R-SLAM",
        default_trajectory_format="tum",
        trajectory_candidates=("poses.txt", "trajectory.txt", "*.tum", "*.txt"),
        pointcloud_candidates=("reconstruction.ply", "map.ply", "*.ply"),
    ),
    "vggt-slam": ExternalSLAMProfile(
        key="vggt-slam",
        display_name="VGGT-SLAM 2.0",
        default_trajectory_format="tum",
        trajectory_candidates=("poses.txt", "trajectory.txt", "*.tum", "*.txt"),
        pointcloud_candidates=("*_points.pcd", "points.pcd", "*.ply", "*.pcd"),
    ),
    "loger": ExternalSLAMProfile(
        key="loger",
        display_name="LoGeR",
        default_trajectory_format="tum",
        trajectory_candidates=("trajectory.txt", "pred_traj.txt", "poses.txt", "predictions.pt", "*.tum", "*.txt"),
        pointcloud_candidates=("points.ply", "*.ply", "*.npy", "*.pcd"),
    ),
    "pi3": ExternalSLAMProfile(
        key="pi3",
        display_name="Pi3/Pi3X",
        default_trajectory_format="tum",
        trajectory_candidates=(
            "poses.txt",
            "trajectory.txt",
            "camera_poses.npz",
            "poses.npz",
            "predictions.pt",
            "*.tum",
            "*.txt",
        ),
        pointcloud_candidates=("result.ply", "points.ply", "*.ply", "*.npy"),
    ),
}

ALIASES = {
    "mast3r": "mast3r-slam",
    "mast3rslam": "mast3r-slam",
    "mast3r_slam": "mast3r-slam",
    "vggt": "vggt-slam",
    "vggt-slam-2": "vggt-slam",
    "vggt-slam-2.0": "vggt-slam",
    "vggt_slam": "vggt-slam",
    "vggt_slam_2": "vggt-slam",
    "vggt_slam_2.0": "vggt-slam",
    "pi3x": "pi3",
}

SYSTEM_CHOICES = tuple(PROFILES)


def normalize_system(system: str | None) -> str:
    """Normalize external system names while keeping CLI choices stable."""

    if not system:
        return "generic"
    key = system.strip().lower().replace(" ", "-")
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        supported = ", ".join(SYSTEM_CHOICES)
        raise ValueError(f"Unsupported external SLAM system '{system}'. Supported: {supported}")
    return key


def resolve_external_slam_artifacts(
    *,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> ExternalSLAMArtifacts:
    """Resolve external SLAM output paths without importing the external project."""

    system_key = normalize_system(system)
    profile = PROFILES[system_key]
    base_dir = _optional_existing_dir(artifact_dir, role="external SLAM output")

    resolved_trajectory = _resolve_required_file(
        explicit=trajectory_path,
        base_dir=base_dir,
        candidates=profile.trajectory_candidates,
        role=f"{profile.display_name} trajectory",
    )
    resolved_pointcloud = _resolve_optional_file(
        explicit=pointcloud_path,
        base_dir=base_dir,
        candidates=profile.pointcloud_candidates,
        role=f"{profile.display_name} point cloud",
    )
    resolved_calib = _resolve_optional_file(
        explicit=pinhole_calib_path,
        base_dir=base_dir,
        candidates=(),
        role="PINHOLE calibration",
    )

    return ExternalSLAMArtifacts(
        system=system_key,
        trajectory_path=resolved_trajectory,
        trajectory_format=trajectory_format or profile.default_trajectory_format,
        pointcloud_path=resolved_pointcloud,
        pinhole_calib_path=resolved_calib,
    )


def import_external_slam(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
    nmea_time_offset_sec: float = 0.0,
) -> str:
    """Convert external visual SLAM artifacts into a COLMAP sparse directory."""

    artifacts = resolve_external_slam_artifacts(
        system=system,
        artifact_dir=artifact_dir,
        trajectory_path=trajectory_path,
        trajectory_format=trajectory_format,
        pointcloud_path=pointcloud_path,
        pinhole_calib_path=pinhole_calib_path,
    )

    trajectory_for_import = artifacts.trajectory_path
    trajectory_format_for_import = artifacts.trajectory_format
    if _is_pose_tensor_artifact(trajectory_for_import):
        trajectory_for_import = materialize_pose_tensor_trajectory(
            trajectory_for_import,
            Path(output_dir) / "external_slam",
        )
        trajectory_format_for_import = "tum"

    from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

    return import_lidar_slam(
        trajectory_path=trajectory_for_import,
        image_dir=image_dir,
        output_dir=output_dir,
        trajectory_format=trajectory_format_for_import,
        pointcloud_path=artifacts.pointcloud_path,
        pinhole_calib_path=artifacts.pinhole_calib_path,
        nmea_time_offset_sec=nmea_time_offset_sec,
    )


def _optional_existing_dir(path: str | Path | None, *, role: str) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{role} directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{role} path is not a directory: {candidate}")
    return candidate


def _resolve_required_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path:
    resolved = _resolve_optional_file(explicit=explicit, base_dir=base_dir, candidates=candidates, role=role)
    if resolved is not None:
        return resolved
    if base_dir is None:
        raise ValueError(f"{role} is required. Pass --trajectory or --external-slam-output.")
    raise FileNotFoundError(f"Could not find {role} under {base_dir}")


def _resolve_optional_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path | None:
    if explicit not in (None, ""):
        return _resolve_explicit_file(Path(explicit), base_dir=base_dir, role=role)
    if base_dir is None:
        return None
    for pattern in candidates:
        matches = _candidate_matches(base_dir, pattern)
        if matches:
            return matches[0]
    return None


def _resolve_explicit_file(path: Path, *, base_dir: Path | None, role: str) -> Path:
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_file():
                raise FileNotFoundError(f"{role} path is not a file: {candidate}")
            return candidate
    raise FileNotFoundError(f"{role} file not found: {path}")


def _candidate_matches(base_dir: Path, pattern: str) -> list[Path]:
    matches = sorted(p for p in base_dir.rglob(pattern) if p.is_file())
    if pattern.endswith(".txt") or pattern == "*.txt":
        matches = [p for p in matches if not _looks_like_non_trajectory_text(p)]
    return matches


def _looks_like_non_trajectory_text(path: Path) -> bool:
    name = path.name.lower()
    return any(skip in name for skip in _SKIP_TEXT_NAMES)


def _is_pose_tensor_artifact(path: Path) -> bool:
    return path.suffix.lower() in _POSE_TENSOR_SUFFIXES


def materialize_pose_tensor_trajectory(trajectory_path: str | Path, output_dir: str | Path) -> Path:
    """Convert a pose tensor artifact into a TUM trajectory file.

    Pi3/Pi3X and LoGeR expose camera-to-world matrices in their Python outputs,
    but their demos often save ``.ply`` / ``.pt`` / ``.npz`` rather than a TUM
    file. Keep that conversion here so downstream training still consumes the
    same simple text trajectory boundary.
    """

    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    poses, timestamps = _load_pose_tensor_artifact(trajectory_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tum_path = output_dir / f"{trajectory_path.stem}_trajectory.tum"
    _write_tum_trajectory(tum_path, poses, timestamps)
    return tum_path


def _load_pose_tensor_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            poses = _extract_pose_matrices(data, role=str(path))
            timestamps = _extract_timestamps(data, len(poses))
            return poses, timestamps
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        data = data.item() if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object else data
        poses = _extract_pose_matrices(data, role=str(path))
        timestamps = _extract_timestamps(data, len(poses))
        return poses, timestamps
    if suffix in (".pt", ".pth"):
        data = _torch_load_cpu(path)
        poses = _extract_pose_matrices(data, role=str(path))
        timestamps = _extract_timestamps(data, len(poses))
        return poses, timestamps
    raise ValueError(f"Unsupported pose tensor artifact: {path}")


def _torch_load_cpu(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is normally available in this project.
        raise ImportError(f"Reading {path.suffix} pose artifacts requires PyTorch to be installed.") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_pose_matrices(data: Any, *, role: str) -> np.ndarray:
    raw = _find_named_value(data, _POSE_MATRIX_KEYS)
    if raw is None:
        raw = data
    poses = _to_numpy(raw)
    poses = _normalize_pose_matrix_shape(poses, role=role)
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"Pose tensor contains non-finite values: {role}")
    return poses


def _extract_timestamps(data: Any, pose_count: int) -> np.ndarray:
    raw = _find_named_value(data, _TIMESTAMP_KEYS)
    if raw is None:
        return np.arange(pose_count, dtype=np.float64)
    timestamps = _to_numpy(raw).astype(np.float64).reshape(-1)
    if len(timestamps) != pose_count:
        raise ValueError(f"Timestamp count {len(timestamps)} does not match pose count {pose_count}")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Pose tensor timestamps contain non-finite values")
    return timestamps


def _find_named_value(data: Any, keys: tuple[str, ...]) -> Any | None:
    if hasattr(data, "files"):
        for key in keys:
            if key in data.files:
                return data[key]
        return None
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        return None
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_pose_matrix_shape(poses: np.ndarray, *, role: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    while poses.ndim > 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, ...]
    if poses.ndim != 3:
        raise ValueError(f"Expected pose tensor with shape (N, 4, 4) or (N, 3, 4), got {poses.shape}: {role}")
    if poses.shape[1:] == (3, 4):
        bottom = np.zeros((poses.shape[0], 1, 4), dtype=np.float64)
        bottom[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=1)
    if poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose tensor with shape (N, 4, 4) or (N, 3, 4), got {poses.shape}: {role}")
    return poses


def _write_tum_trajectory(path: Path, poses: np.ndarray, timestamps: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, pose in zip(timestamps, poses):
            qw, qx, qy, qz = _rotation_to_quaternion(pose[:3, :3])
            tx, ty, tz = pose[:3, 3]
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[2, 1] - rotation[1, 2]) * scale
        qy = (rotation[0, 2] - rotation[2, 0]) * scale
        qz = (rotation[1, 0] - rotation[0, 1]) * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("Rotation matrix produced a zero-length quaternion")
    return qw / norm, qx / norm, qy / norm, qz / norm
