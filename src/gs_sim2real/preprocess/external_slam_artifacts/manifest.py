"""Build lightweight manifests for external SLAM artifact imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gs_sim2real.preprocess.external_slam_artifacts.point_tensor import is_point_tensor_artifact
from gs_sim2real.preprocess.external_slam_artifacts.pose_tensor import is_pose_tensor_artifact
from gs_sim2real.preprocess.external_slam_artifacts.profiles import PROFILES
from gs_sim2real.preprocess.external_slam_artifacts.resolver import resolve_external_slam_artifacts


def build_external_slam_artifact_manifest(
    *,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve external SLAM artifacts and describe how they will be imported."""

    artifacts = resolve_external_slam_artifacts(
        system=system,
        artifact_dir=artifact_dir,
        trajectory_path=trajectory_path,
        trajectory_format=trajectory_format,
        pointcloud_path=pointcloud_path,
        pinhole_calib_path=pinhole_calib_path,
    )
    profile = PROFILES[artifacts.system]
    return {
        "type": "external-slam-artifact-manifest",
        "system": artifacts.system,
        "displayName": profile.display_name,
        "artifactDir": str(Path(artifact_dir)) if artifact_dir not in (None, "") else None,
        "trajectory": _summarize_trajectory(artifacts.trajectory_path, artifacts.trajectory_format),
        "pointcloud": _summarize_pointcloud(artifacts.pointcloud_path),
        "pinholeCalibration": _summarize_optional_file(artifacts.pinhole_calib_path, role="pinhole_calibration"),
        "ready": True,
    }


def render_external_slam_artifact_manifest_text(manifest: dict[str, Any]) -> str:
    """Render a manifest for command-line dry runs."""

    lines = [
        f"External SLAM artifacts: {manifest['displayName']} ({manifest['system']})",
        f"- trajectory: {_format_artifact(manifest['trajectory'])}",
        f"- point cloud: {_format_artifact(manifest['pointcloud'])}",
        f"- pinhole calibration: {_format_artifact(manifest['pinholeCalibration'])}",
    ]
    return "\n".join(lines) + "\n"


def render_external_slam_artifact_manifest_json(manifest: dict[str, Any]) -> str:
    """Render a manifest as stable JSON."""

    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _summarize_trajectory(path: Path, trajectory_format: str) -> dict[str, Any]:
    materialization = "pose_tensor_to_tum" if is_pose_tensor_artifact(path) else "direct"
    return {
        **_summarize_file(path, role="trajectory"),
        "format": trajectory_format,
        "materialization": materialization,
    }


def _summarize_pointcloud(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    materialization = "point_tensor_to_npy" if is_point_tensor_artifact(path) else "direct"
    return {
        **_summarize_file(path, role="pointcloud"),
        "materialization": materialization,
    }


def _summarize_optional_file(path: Path | None, *, role: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return _summarize_file(path, role=role)


def _summarize_file(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
    }


def _format_artifact(artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "n/a"
    parts = [
        artifact["path"],
        _format_bytes(artifact["bytes"]),
    ]
    if "format" in artifact:
        parts.append(f"format={artifact['format']}")
    if "materialization" in artifact:
        parts.append(f"materialization={artifact['materialization']}")
    return " (" + ", ".join(parts[1:]) + ")" if not parts[0] else f"{parts[0]} ({', '.join(parts[1:])})"


def _format_bytes(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value} B"


__all__ = [
    "build_external_slam_artifact_manifest",
    "render_external_slam_artifact_manifest_json",
    "render_external_slam_artifact_manifest_text",
]
