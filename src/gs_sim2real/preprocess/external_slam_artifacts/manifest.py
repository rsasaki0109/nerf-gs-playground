"""Build lightweight manifests for external SLAM artifact imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.preprocess.external_slam_artifacts.point_tensor import (
    inspect_point_tensor_artifact,
    is_point_tensor_artifact,
)
from gs_sim2real.preprocess.external_slam_artifacts.pose_tensor import (
    inspect_pose_tensor_artifact,
    is_pose_tensor_artifact,
)
from gs_sim2real.preprocess.external_slam_artifacts.profiles import PROFILES, ExternalSLAMProfile, normalize_system
from gs_sim2real.preprocess.external_slam_artifacts.resolver import (
    ExternalSLAMCandidateTrace,
    ExternalSLAMFileResolutionTrace,
    resolve_external_slam_artifacts,
    trace_external_slam_file_resolution,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_external_slam_artifact_manifest(
    *,
    image_dir: str | Path | None = None,
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
    image_summary = _summarize_images(image_dir)
    trajectory_summary = _summarize_trajectory(artifacts.trajectory_path, artifacts.trajectory_format)
    pointcloud_summary = _summarize_pointcloud(artifacts.pointcloud_path)
    return {
        "type": "external-slam-artifact-manifest",
        "system": artifacts.system,
        "displayName": profile.display_name,
        "artifactDir": str(Path(artifact_dir)) if artifact_dir not in (None, "") else None,
        "resolution": _build_resolution_summary(
            profile=profile,
            artifact_dir=artifact_dir,
            trajectory_path=trajectory_path,
            trajectory_format=artifacts.trajectory_format,
            pointcloud_path=pointcloud_path,
            pinhole_calib_path=pinhole_calib_path,
            resolved_trajectory=artifacts.trajectory_path,
            resolved_pointcloud=artifacts.pointcloud_path,
            resolved_calib=artifacts.pinhole_calib_path,
        ),
        "images": image_summary,
        "trajectory": trajectory_summary,
        "pointcloud": pointcloud_summary,
        "pinholeCalibration": _summarize_optional_file(artifacts.pinhole_calib_path, role="pinhole_calibration"),
        "alignment": _summarize_alignment(image_summary, trajectory_summary),
        "ready": True,
    }


def build_external_slam_artifact_error_manifest(
    *,
    error: BaseException,
    image_dir: str | Path | None = None,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable dry-run manifest when artifact resolution fails."""

    system_key, profile = _safe_profile(system)
    image_summary = _summarize_images(image_dir)
    return {
        "type": "external-slam-artifact-manifest",
        "system": system_key,
        "displayName": profile.display_name if profile else "external SLAM",
        "artifactDir": str(Path(artifact_dir)) if artifact_dir not in (None, "") else None,
        "resolution": _build_resolution_summary(
            profile=profile,
            artifact_dir=artifact_dir,
            trajectory_path=trajectory_path,
            trajectory_format=trajectory_format or (profile.default_trajectory_format if profile else None),
            pointcloud_path=pointcloud_path,
            pinhole_calib_path=pinhole_calib_path,
            resolved_trajectory=None,
            resolved_pointcloud=None,
            resolved_calib=None,
        ),
        "images": image_summary,
        "trajectory": None,
        "pointcloud": None,
        "pinholeCalibration": None,
        "alignment": _summarize_alignment(image_summary, {}),
        "ready": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def render_external_slam_artifact_manifest_text(manifest: dict[str, Any]) -> str:
    """Render a manifest for command-line dry runs."""

    lines = [
        f"External SLAM artifacts: {manifest['displayName']} ({manifest['system']})",
        f"- images: {_format_images(manifest['images'])}",
        f"- trajectory: {_format_artifact(manifest['trajectory'])}",
        f"- point cloud: {_format_artifact(manifest['pointcloud'])}",
        f"- pinhole calibration: {_format_artifact(manifest['pinholeCalibration'])}",
        f"- alignment: {_format_alignment(manifest['alignment'])}",
    ]
    if manifest.get("error"):
        lines.append(f"- error: {manifest['error']['message']}")
    return "\n".join(lines) + "\n"


def render_external_slam_artifact_manifest_json(manifest: dict[str, Any]) -> str:
    """Render a manifest as stable JSON."""

    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _summarize_trajectory(path: Path, trajectory_format: str) -> dict[str, Any]:
    materialization = "pose_tensor_to_tum" if is_pose_tensor_artifact(path) else "direct"
    summary = {
        **_summarize_file(path, role="trajectory"),
        "format": trajectory_format,
        "materialization": materialization,
    }
    if is_pose_tensor_artifact(path):
        summary.update(inspect_pose_tensor_artifact(path))
    else:
        summary["poseCount"] = _count_text_trajectory_rows(path, trajectory_format)
    return summary


def _summarize_pointcloud(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    materialization = "point_tensor_to_npy" if is_point_tensor_artifact(path) else "direct"
    summary = {
        **_summarize_file(path, role="pointcloud"),
        "materialization": materialization,
    }
    if is_point_tensor_artifact(path):
        summary.update(inspect_point_tensor_artifact(path))
    elif path.suffix.lower() == ".ply":
        summary["pointCount"] = _read_ply_vertex_count(path)
    elif path.suffix.lower() == ".npy":
        summary["pointCount"] = _read_npy_point_count(path)
    return summary


def _summarize_images(image_dir: str | Path | None) -> dict[str, Any] | None:
    if image_dir in (None, ""):
        return None
    path = Path(image_dir)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "isDirectory": path.is_dir(),
        "imageCount": None,
    }
    if path.is_dir():
        summary["imageCount"] = sum(
            1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
        )
    return summary


def _summarize_alignment(images: dict[str, Any] | None, trajectory: dict[str, Any]) -> dict[str, Any]:
    image_count = images.get("imageCount") if images else None
    pose_count = trajectory.get("poseCount")
    summary: dict[str, Any] = {
        "strategy": "sequential_count_check",
        "imageCount": image_count,
        "trajectoryPoseCount": pose_count,
        "alignedFrameCount": None,
        "droppedImageCount": None,
        "unusedPoseCount": None,
        "status": "unknown",
        "message": "image or trajectory pose count is unavailable",
    }
    if image_count is None or pose_count is None:
        return summary

    aligned_count = min(int(image_count), int(pose_count))
    dropped_count = max(0, int(image_count) - int(pose_count))
    unused_count = max(0, int(pose_count) - int(image_count))
    status = "ok"
    message = "image and pose counts match"
    if aligned_count < 2:
        status = "error"
        message = "fewer than 2 frames would be imported"
    elif dropped_count or unused_count:
        status = "warning"
        message = "image and pose counts differ"

    summary.update(
        {
            "alignedFrameCount": aligned_count,
            "droppedImageCount": dropped_count,
            "unusedPoseCount": unused_count,
            "status": status,
            "message": message,
        }
    )
    return summary


def _safe_profile(system: str | None) -> tuple[str, ExternalSLAMProfile | None]:
    try:
        system_key = normalize_system(system)
    except ValueError:
        return system or "generic", None
    return system_key, PROFILES[system_key]


def _build_resolution_summary(
    *,
    profile: ExternalSLAMProfile | None,
    artifact_dir: str | Path | None,
    trajectory_path: str | Path | None,
    trajectory_format: str | None,
    pointcloud_path: str | Path | None,
    pinhole_calib_path: str | Path | None,
    resolved_trajectory: Path | None,
    resolved_pointcloud: Path | None,
    resolved_calib: Path | None,
) -> dict[str, Any]:
    base_dir = str(Path(artifact_dir)) if artifact_dir not in (None, "") else None
    trajectory_candidates = profile.trajectory_candidates if profile else ()
    pointcloud_candidates = profile.pointcloud_candidates if profile else ()
    trajectory_trace = trace_external_slam_file_resolution(
        explicit=trajectory_path,
        base_dir=artifact_dir,
        candidates=trajectory_candidates,
        role="trajectory",
    )
    pointcloud_trace = trace_external_slam_file_resolution(
        explicit=pointcloud_path,
        base_dir=artifact_dir,
        candidates=pointcloud_candidates,
        role="pointcloud",
    )
    calib_trace = trace_external_slam_file_resolution(
        explicit=pinhole_calib_path,
        base_dir=artifact_dir,
        candidates=(),
        role="pinhole_calibration",
    )
    return {
        "trajectory": _resolution_section(
            role="trajectory",
            explicit_path=trajectory_path,
            base_dir=base_dir,
            candidate_patterns=trajectory_candidates,
            selected_path=resolved_trajectory,
            trajectory_format=trajectory_format,
            trace=trajectory_trace,
        ),
        "pointcloud": _resolution_section(
            role="pointcloud",
            explicit_path=pointcloud_path,
            base_dir=base_dir,
            candidate_patterns=pointcloud_candidates,
            selected_path=resolved_pointcloud,
            trace=pointcloud_trace,
        ),
        "pinholeCalibration": _resolution_section(
            role="pinhole_calibration",
            explicit_path=pinhole_calib_path,
            base_dir=base_dir,
            candidate_patterns=(),
            selected_path=resolved_calib,
            trace=calib_trace,
        ),
    }


def _resolution_section(
    *,
    role: str,
    explicit_path: str | Path | None,
    base_dir: str | None,
    candidate_patterns: tuple[str, ...],
    selected_path: Path | None,
    trajectory_format: str | None = None,
    trace: ExternalSLAMFileResolutionTrace | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "role": role,
        "explicitPath": str(Path(explicit_path)) if explicit_path not in (None, "") else None,
        "baseDir": base_dir,
        "candidatePatterns": list(candidate_patterns),
        "selectedPath": str(selected_path) if selected_path is not None else None,
    }
    if trajectory_format is not None:
        summary["format"] = trajectory_format
    if trace is not None:
        summary["reason"] = trace.reason
        summary["trace"] = [_candidate_trace_to_dict(item) for item in trace.candidate_traces]
    return summary


def _candidate_trace_to_dict(trace: ExternalSLAMCandidateTrace) -> dict[str, Any]:
    return {
        "pattern": trace.pattern,
        "matchCount": trace.match_count,
        "selectedPath": str(trace.selected_path) if trace.selected_path is not None else None,
        "skippedPaths": [str(path) for path in trace.skipped_paths],
        "reason": trace.reason,
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


def _count_text_trajectory_rows(path: Path, trajectory_format: str) -> int | None:
    if trajectory_format == "nmea":
        return None
    min_columns = 12 if trajectory_format == "kitti" else 8
    count = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if len(text.split()) >= min_columns:
                count += 1
    return count


def _read_ply_vertex_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            for _ in range(256):
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except (OSError, ValueError):
        return None
    return None


def _read_npy_point_count(path: Path) -> int | None:
    try:
        points = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if points.ndim < 2:
        return None
    return int(points.reshape(-1, points.shape[-1]).shape[0])


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
    if "poseCount" in artifact and artifact["poseCount"] is not None:
        parts.append(f"poses={artifact['poseCount']}")
    if "pointCount" in artifact and artifact["pointCount"] is not None:
        parts.append(f"points={artifact['pointCount']}")
    return " (" + ", ".join(parts[1:]) + ")" if not parts[0] else f"{parts[0]} ({', '.join(parts[1:])})"


def _format_images(images: dict[str, Any] | None) -> str:
    if images is None:
        return "n/a"
    if not images["exists"]:
        return f"{images['path']} (not found)"
    if not images["isDirectory"]:
        return f"{images['path']} (not a directory)"
    return f"{images['path']} ({images['imageCount']} images)"


def _format_alignment(alignment: dict[str, Any]) -> str:
    if alignment["status"] == "unknown":
        return "unknown"
    return (
        f"{alignment['status']} ({alignment['alignedFrameCount']} aligned, "
        f"{alignment['droppedImageCount']} dropped images, {alignment['unusedPoseCount']} unused poses)"
    )


def _format_bytes(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value} B"


__all__ = [
    "build_external_slam_artifact_error_manifest",
    "build_external_slam_artifact_manifest",
    "render_external_slam_artifact_manifest_json",
    "render_external_slam_artifact_manifest_text",
]
