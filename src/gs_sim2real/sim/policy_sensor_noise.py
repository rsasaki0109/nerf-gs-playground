"""Sensor noise profiles for route policy scenario benchmarks.

The route policy gym adapter surfaces the observed pose and goal as part of
the feature dict that feeds the policy. Real robotics stacks see those
quantities through sensors with non-zero uncertainty, so a benchmark that
wants to compare policies under realistic noise needs a seam that perturbs
the observed pose without touching the true simulator state.

This module defines a small, JSON-serialisable ``RoutePolicySensorNoiseProfile``
record plus a deterministic helper that applies Gaussian noise to a
:class:`~gs_sim2real.sim.interfaces.Pose3D`. The noise is always driven by a
seed derived from ``(reset_seed, episode_index, step_index, profile_id, kind)``
with ``hashlib.sha256``, so the same scenario replay produces identical noisy
observations regardless of Python's hash randomization.

Only additive presentation: the adapter's *true* pose (used for collision
checks, trajectory scoring, route rollouts) is never mutated here — only the
pose the policy is allowed to observe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from .interfaces import Pose3D


ROUTE_POLICY_SENSOR_NOISE_PROFILE_VERSION = "gs-mapper-route-policy-sensor-noise-profile/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicySensorNoiseProfile:
    """Gaussian noise budget for the observed pose / goal used by a policy.

    All ``*_std`` values are interpreted as the standard deviation of a
    zero-mean Gaussian applied additively to the respective quantity. A
    value of ``0.0`` disables that axis.
    """

    profile_id: str
    pose_position_std_meters: float = 0.0
    pose_heading_std_radians: float = 0.0
    goal_position_std_meters: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SENSOR_NOISE_PROFILE_VERSION

    def __post_init__(self) -> None:
        if not str(self.profile_id):
            raise ValueError("profile_id must not be empty")
        _non_negative_float(self.pose_position_std_meters, "pose_position_std_meters")
        _non_negative_float(self.pose_heading_std_radians, "pose_heading_std_radians")
        _non_negative_float(self.goal_position_std_meters, "goal_position_std_meters")

    @property
    def is_noise_free(self) -> bool:
        return (
            self.pose_position_std_meters == 0.0
            and self.pose_heading_std_radians == 0.0
            and self.goal_position_std_meters == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-sensor-noise-profile",
            "version": self.version,
            "profileId": self.profile_id,
            "posePositionStdMeters": float(self.pose_position_std_meters),
            "poseHeadingStdRadians": float(self.pose_heading_std_radians),
            "goalPositionStdMeters": float(self.goal_position_std_meters),
            "metadata": _json_mapping(self.metadata),
        }


def write_route_policy_sensor_noise_profile_json(
    path: str | Path,
    profile: RoutePolicySensorNoiseProfile,
) -> Path:
    """Persist a sensor noise profile as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_sensor_noise_profile_json(path: str | Path) -> RoutePolicySensorNoiseProfile:
    """Load a sensor noise profile JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_sensor_noise_profile_from_dict(_mapping(payload, "sensorNoiseProfile"))


def route_policy_sensor_noise_profile_from_dict(payload: Mapping[str, Any]) -> RoutePolicySensorNoiseProfile:
    """Rebuild a sensor noise profile from JSON."""

    _record_type(payload, "route-policy-sensor-noise-profile")
    version = str(payload.get("version", ROUTE_POLICY_SENSOR_NOISE_PROFILE_VERSION))
    if version != ROUTE_POLICY_SENSOR_NOISE_PROFILE_VERSION:
        raise ValueError(f"unsupported route policy sensor noise profile version: {version}")
    return RoutePolicySensorNoiseProfile(
        profile_id=str(payload["profileId"]),
        pose_position_std_meters=float(payload.get("posePositionStdMeters", 0.0)),
        pose_heading_std_radians=float(payload.get("poseHeadingStdRadians", 0.0)),
        goal_position_std_meters=float(payload.get("goalPositionStdMeters", 0.0)),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_sensor_noise_profile_markdown(profile: RoutePolicySensorNoiseProfile) -> str:
    """Render a compact Markdown summary for a sensor noise profile."""

    lines = [
        f"# Route Policy Sensor Noise Profile: {profile.profile_id}",
        f"- Pose position σ (m): {profile.pose_position_std_meters}",
        f"- Pose heading σ (rad): {profile.pose_heading_std_radians}",
        f"- Goal position σ (m): {profile.goal_position_std_meters}",
        f"- Noise free: {'yes' if profile.is_noise_free else 'no'}",
    ]
    if profile.metadata:
        lines.append("")
        lines.append("| Metadata | Value |")
        lines.append("| --- | --- |")
        for key in sorted(profile.metadata):
            lines.append(f"| {key} | {profile.metadata[key]} |")
    return "\n".join(lines) + "\n"


def sensor_noise_rng(
    *,
    base_seed: int | None,
    profile_id: str,
    episode_index: int,
    step_index: int,
    kind: str,
) -> random.Random:
    """Return a ``random.Random`` seeded from the noise context.

    The seed is derived via SHA-256 over the context tuple so replaying a
    scenario with the same base seed produces bit-identical noise even across
    Python interpreter restarts (Python's built-in ``hash`` is randomised by
    default).
    """

    resolved_base = "none" if base_seed is None else str(int(base_seed))
    digest = hashlib.sha256(
        f"{resolved_base}|{profile_id}|{int(episode_index)}|{int(step_index)}|{kind}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def apply_sensor_noise_to_pose(
    pose: Pose3D,
    profile: RoutePolicySensorNoiseProfile,
    *,
    rng: random.Random,
    perturb_position: bool = True,
    perturb_heading: bool = True,
    position_std_override: float | None = None,
) -> Pose3D:
    """Return a copy of ``pose`` with Gaussian noise added per ``profile``.

    ``position_std_override`` lets callers reuse the same profile for both
    the observed pose (σ = ``pose_position_std_meters``) and the observed
    goal (σ = ``goal_position_std_meters``) without constructing a second
    profile. When the chosen axis has σ == 0 the pose is returned unchanged
    along that axis so profiles with ``is_noise_free`` stay identity.
    """

    position_std = (
        float(position_std_override) if position_std_override is not None else float(profile.pose_position_std_meters)
    )
    heading_std = float(profile.pose_heading_std_radians)

    position = pose.position
    if perturb_position and position_std > 0.0:
        delta = tuple(rng.gauss(0.0, position_std) for _ in range(3))
        position = (
            float(position[0] + delta[0]),
            float(position[1] + delta[1]),
            float(position[2] + delta[2]),
        )
    orientation = pose.orientation_xyzw
    if perturb_heading and heading_std > 0.0:
        yaw_delta = float(rng.gauss(0.0, heading_std))
        orientation = _rotate_quaternion_yaw(orientation, yaw_delta)
    return Pose3D(
        position=position,
        orientation_xyzw=orientation,
        frame_id=pose.frame_id,
        timestamp_seconds=pose.timestamp_seconds,
    )


def _rotate_quaternion_yaw(
    orientation: tuple[float, float, float, float],
    yaw_delta_radians: float,
) -> tuple[float, float, float, float]:
    """Compose ``orientation`` (xyzw) with a yaw rotation around +Z."""

    half = 0.5 * float(yaw_delta_radians)
    qz = math.sin(half)
    qw = math.cos(half)
    x1, y1, z1, w1 = (float(c) for c in orientation)
    # Hamilton product: q_yaw * q_orig with q_yaw = (0, 0, qz, qw).
    nx = qw * x1 - qz * y1
    ny = qw * y1 + qz * x1
    nz = qw * z1 + qz * w1
    nw = qw * w1 - qz * z1
    norm = math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    if norm <= 0.0:
        return orientation
    return (nx / norm, ny / norm, nz / norm, nw / norm)


def _non_negative_float(value: float, field_name: str) -> None:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")
