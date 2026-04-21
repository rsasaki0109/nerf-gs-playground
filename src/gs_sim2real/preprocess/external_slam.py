"""Import artifacts produced by external visual SLAM front-ends.

This module intentionally does not import MASt3R-SLAM, VGGT-SLAM, LoGeR,
Pi3, or their model dependencies. It only normalizes their exported trajectory
and optional point-cloud files into the existing COLMAP text import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        trajectory_candidates=("trajectory.txt", "poses.txt", "*.tum", "*.txt"),
        pointcloud_candidates=("points.ply", "*.ply", "*.npy", "*.pcd"),
    ),
    "pi3": ExternalSLAMProfile(
        key="pi3",
        display_name="Pi3/Pi3X",
        default_trajectory_format="tum",
        trajectory_candidates=("poses.txt", "trajectory.txt", "*.tum", "*.txt"),
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

    from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

    return import_lidar_slam(
        trajectory_path=artifacts.trajectory_path,
        image_dir=image_dir,
        output_dir=output_dir,
        trajectory_format=artifacts.trajectory_format,
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
