"""Artifact-only integration boundary for external visual SLAM front-ends."""

from __future__ import annotations

from gs_sim2real.preprocess.external_slam_artifacts.importer import import_external_slam
from gs_sim2real.preprocess.external_slam_artifacts.manifest import (
    build_external_slam_artifact_manifest,
    render_external_slam_artifact_manifest_json,
    render_external_slam_artifact_manifest_text,
)
from gs_sim2real.preprocess.external_slam_artifacts.pose_tensor import materialize_pose_tensor_trajectory
from gs_sim2real.preprocess.external_slam_artifacts.profiles import (
    ALIASES,
    PROFILES,
    SYSTEM_CHOICES,
    ExternalSLAMProfile,
    normalize_system,
)
from gs_sim2real.preprocess.external_slam_artifacts.resolver import (
    ExternalSLAMArtifacts,
    resolve_external_slam_artifacts,
)

__all__ = [
    "ALIASES",
    "PROFILES",
    "SYSTEM_CHOICES",
    "ExternalSLAMArtifacts",
    "ExternalSLAMProfile",
    "build_external_slam_artifact_manifest",
    "import_external_slam",
    "materialize_pose_tensor_trajectory",
    "normalize_system",
    "render_external_slam_artifact_manifest_json",
    "render_external_slam_artifact_manifest_text",
    "resolve_external_slam_artifacts",
]
