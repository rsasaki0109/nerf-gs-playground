"""Tests for route policy sensor noise profiles."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicySensorNoiseProfile,
    apply_sensor_noise_to_pose,
    build_simulation_catalog,
    load_route_policy_sensor_noise_profile_json,
    render_route_policy_sensor_noise_profile_markdown,
    route_policy_sensor_noise_profile_from_dict,
    sensor_noise_rng,
    write_route_policy_sensor_noise_profile_json,
)


def _unit_catalog() -> object:
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )


def _unit_pose(position: tuple[float, float, float] = (0.1, -0.2, 0.05)) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0))


def test_sensor_noise_profile_round_trips_through_json(tmp_path: Path) -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="unit-noise",
        pose_position_std_meters=0.05,
        pose_heading_std_radians=0.01,
        goal_position_std_meters=0.02,
        metadata={"sensor": "gnss+imu"},
    )
    path = write_route_policy_sensor_noise_profile_json(tmp_path / "noise.json", profile)
    loaded = load_route_policy_sensor_noise_profile_json(path)

    assert loaded == profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = route_policy_sensor_noise_profile_from_dict(payload)
    assert rebuilt == profile


def test_sensor_noise_profile_rejects_negative_std() -> None:
    with pytest.raises(ValueError):
        RoutePolicySensorNoiseProfile(profile_id="bad", pose_position_std_meters=-0.1)
    with pytest.raises(ValueError):
        RoutePolicySensorNoiseProfile(profile_id="bad", pose_heading_std_radians=float("nan"))


def test_sensor_noise_profile_is_noise_free_when_all_stds_zero() -> None:
    zero = RoutePolicySensorNoiseProfile(profile_id="zero")
    assert zero.is_noise_free is True

    noisy = RoutePolicySensorNoiseProfile(profile_id="noisy", pose_position_std_meters=0.01)
    assert noisy.is_noise_free is False


def test_apply_sensor_noise_is_identity_when_profile_is_noise_free() -> None:
    profile = RoutePolicySensorNoiseProfile(profile_id="zero")
    pose = _unit_pose()
    rng = sensor_noise_rng(
        base_seed=42,
        profile_id=profile.profile_id,
        episode_index=0,
        step_index=0,
        kind="pose",
    )
    assert apply_sensor_noise_to_pose(pose, profile, rng=rng) == pose


def test_apply_sensor_noise_is_deterministic_across_invocations() -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="unit-noise",
        pose_position_std_meters=0.10,
        pose_heading_std_radians=0.05,
    )
    pose = _unit_pose()
    rng_a = sensor_noise_rng(base_seed=2026, profile_id=profile.profile_id, episode_index=1, step_index=3, kind="pose")
    rng_b = sensor_noise_rng(base_seed=2026, profile_id=profile.profile_id, episode_index=1, step_index=3, kind="pose")
    noisy_a = apply_sensor_noise_to_pose(pose, profile, rng=rng_a)
    noisy_b = apply_sensor_noise_to_pose(pose, profile, rng=rng_b)
    assert noisy_a == noisy_b
    # The noisy pose must actually differ from the input for non-zero noise.
    assert noisy_a.position != pose.position
    # Orientation stays a unit quaternion.
    length_squared = sum(component * component for component in noisy_a.orientation_xyzw)
    assert math.isclose(length_squared, 1.0, rel_tol=1e-6)


def test_apply_sensor_noise_differs_with_different_contexts() -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="unit-noise",
        pose_position_std_meters=0.25,
    )
    pose = _unit_pose()
    baseline = apply_sensor_noise_to_pose(
        pose,
        profile,
        rng=sensor_noise_rng(base_seed=1, profile_id=profile.profile_id, episode_index=0, step_index=0, kind="pose"),
    )
    # Changing episode or step must move the noise.
    different_step = apply_sensor_noise_to_pose(
        pose,
        profile,
        rng=sensor_noise_rng(base_seed=1, profile_id=profile.profile_id, episode_index=0, step_index=1, kind="pose"),
    )
    different_episode = apply_sensor_noise_to_pose(
        pose,
        profile,
        rng=sensor_noise_rng(base_seed=1, profile_id=profile.profile_id, episode_index=1, step_index=0, kind="pose"),
    )
    different_base = apply_sensor_noise_to_pose(
        pose,
        profile,
        rng=sensor_noise_rng(base_seed=2, profile_id=profile.profile_id, episode_index=0, step_index=0, kind="pose"),
    )
    assert baseline.position != different_step.position
    assert baseline.position != different_episode.position
    assert baseline.position != different_base.position


def test_apply_sensor_noise_goal_override_scales_deltas() -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="scale-noise",
        pose_position_std_meters=0.10,
        goal_position_std_meters=1.50,
    )
    pose = _unit_pose()
    pose_rng = sensor_noise_rng(base_seed=7, profile_id=profile.profile_id, episode_index=0, step_index=0, kind="pose")
    goal_rng = sensor_noise_rng(base_seed=7, profile_id=profile.profile_id, episode_index=0, step_index=0, kind="goal")
    pose_noisy = apply_sensor_noise_to_pose(pose, profile, rng=pose_rng)
    goal_noisy = apply_sensor_noise_to_pose(
        pose,
        profile,
        rng=goal_rng,
        perturb_heading=False,
        position_std_override=profile.goal_position_std_meters,
    )
    pose_delta = math.dist(pose.position, pose_noisy.position)
    goal_delta = math.dist(pose.position, goal_noisy.position)
    # Same base seed but different "kind" → different RNG → different deltas.
    assert pose_noisy.position != goal_noisy.position
    # Goal noise has a σ ~15x larger than pose noise, so the magnitude should
    # typically scale with it. Guard loosely — absolute deltas are random but
    # with the same seeds the expected sign of the inequality holds for this
    # specific fixture.
    assert goal_delta > pose_delta


def test_gym_adapter_without_noise_profile_feature_parity() -> None:
    adapter = _build_adapter(noise_profile=None)
    observation, _ = adapter.reset(seed=123, goal=(0.3, 0.0, 0.0))

    # Feature vector still contains the canonical keys.
    assert "goal-distance-meters" in observation
    assert "pose-position-x" in observation
    # Pose matches the true state — the adapter state pose equals the observed
    # pose feature for the identity noise profile.
    state = adapter.state
    assert math.isclose(observation["pose-position-x"], state.pose.position[0], rel_tol=0, abs_tol=1e-9)


def test_gym_adapter_with_noise_profile_perturbs_features_deterministically() -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="adapter-noise",
        pose_position_std_meters=0.25,
        pose_heading_std_radians=0.05,
        goal_position_std_meters=0.30,
    )
    adapter_noisy_a = _build_adapter(noise_profile=profile)
    adapter_noisy_b = _build_adapter(noise_profile=profile)
    adapter_clean = _build_adapter(noise_profile=None)

    obs_noisy_a, _ = adapter_noisy_a.reset(seed=7, goal=(0.4, 0.0, 0.0))
    obs_noisy_b, _ = adapter_noisy_b.reset(seed=7, goal=(0.4, 0.0, 0.0))
    obs_clean, _ = adapter_clean.reset(seed=7, goal=(0.4, 0.0, 0.0))

    # Deterministic given the same profile + seed.
    assert obs_noisy_a == obs_noisy_b
    # But different from the noise-free baseline.
    assert obs_noisy_a["pose-position-x"] != obs_clean["pose-position-x"]
    assert obs_noisy_a["goal-distance-meters"] != obs_clean["goal-distance-meters"]


def test_sensor_noise_profile_markdown_lists_all_knobs() -> None:
    profile = RoutePolicySensorNoiseProfile(
        profile_id="rendered-noise",
        pose_position_std_meters=0.10,
        pose_heading_std_radians=0.02,
        goal_position_std_meters=0.25,
        metadata={"family": "gnss-outdoor"},
    )
    text = render_route_policy_sensor_noise_profile_markdown(profile)

    assert "Route Policy Sensor Noise Profile: rendered-noise" in text
    assert "Pose position σ (m): 0.1" in text
    assert "Pose heading σ (rad): 0.02" in text
    assert "Goal position σ (m): 0.25" in text
    assert "| family | gnss-outdoor |" in text


def _build_adapter(*, noise_profile: RoutePolicySensorNoiseProfile | None) -> RoutePolicyGymAdapter:
    env = HeadlessPhysicalAIEnvironment(_unit_catalog())
    return RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            max_steps=4,
            goal_tolerance_meters=0.05,
            sensor_noise_profile=noise_profile,
        ),
    )
