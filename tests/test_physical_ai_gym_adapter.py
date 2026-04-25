"""Tests for Gymnasium-style route policy adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RouteCandidate,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RouteRewardWeights,
    build_simulation_catalog,
    make_route_policy_env,
)


def test_route_policy_gym_reset_returns_observation_and_info() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(env, RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4))
    goal = unit_pose((0.5, 0.0, 0.0))

    observation, info = adapter.reset(goal=goal)

    assert info["sceneId"] == "unit-scene"
    assert info["stepIndex"] == 0
    assert info["goal"]["position"] == [0.5, 0.0, 0.0]
    assert observation["episode-step-index"] == 0.0
    assert observation["remaining-step-fraction"] == 1.0
    assert observation["goal-distance-meters"] == pytest.approx(0.5)
    assert observation["goal-delta-x"] == pytest.approx(0.5)


def test_route_policy_gym_step_executes_target_action_without_ending_before_goal() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            goal_reward=2.0,
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
        ),
    )
    adapter.reset(goal=unit_pose((0.75, 0.0, 0.0)))

    observation, reward, terminated, truncated, info = adapter.step(
        {"routeId": "halfway", "target": {"x": 0.25, "y": 0.0, "z": 0.0}}
    )

    assert terminated is False
    assert truncated is False
    assert info["termination_reason"] is None
    assert info["policySample"]["observation"]["sourceType"] == "route-rollout"
    assert info["policySample"]["observation"]["sourceId"] == "halfway"
    assert observation["route-passed"] == 1.0
    assert observation["route-applied-step-count"] == 1.0
    assert observation["goal-distance-meters"] == pytest.approx(0.5)
    assert reward == pytest.approx(info["policySample"]["reward"]["reward"])


def test_route_policy_gym_step_adds_goal_reward_and_terminates_at_goal() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = make_route_policy_env(
        env,
        scene_id="unit-scene",
        goal_reward=2.0,
        reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
    )
    adapter.reset(goal=unit_pose((0.25, 0.0, 0.0)))

    observation, reward, terminated, truncated, info = adapter.step((0.25, 0.0, 0.0))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "goal-reached"
    assert info["goal_reached"] is True
    assert observation["goal-reached"] == 1.0
    assert reward == pytest.approx(info["policySample"]["reward"]["reward"] + 2.0)


def test_route_policy_gym_step_terminates_blocked_route_with_policy_sample() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
        ),
    )
    adapter.reset(goal=unit_pose((0.5, 0.0, 0.0)))

    observation, reward, terminated, truncated, info = adapter.step({"target": {"x": 2.0, "y": 0.0, "z": 0.0}})

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "blocked-route"
    assert info["blocked"] is True
    assert info["policySample"]["reward"]["passed"] is False
    assert observation["route-blocked-step-index"] == 0.0
    assert observation["route-collision-count"] == 1.0
    assert reward == pytest.approx(-1.0)


def test_route_policy_gym_step_truncates_at_max_steps() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            max_steps=1,
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
            truncation_penalty=-0.25,
        ),
    )
    adapter.reset(goal=unit_pose((0.75, 0.0, 0.0)))

    _, reward, terminated, truncated, info = adapter.step({"target": {"x": 0.25, "y": 0.0, "z": 0.0}})

    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "max-steps"
    assert info["done"] is True
    assert reward == pytest.approx(info["policySample"]["reward"]["reward"] - 0.25)


def test_route_policy_gym_action_parser_anchors_candidate_routes() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(env, RoutePolicyEnvConfig(scene_id="unit-scene"))
    adapter.reset(goal=unit_pose((0.5, 0.0, 0.0)))

    _, _, _, _, info = adapter.step(RouteCandidate("candidate-route", (unit_pose((0.25, 0.0, 0.0)),)))

    route = info["route"]
    assert route["routeId"] == "candidate-route"
    assert route["trajectory"][0]["position"] == [0.0, 0.0, 0.0]
    assert route["trajectory"][1]["position"] == [0.25, 0.0, 0.0]


def test_route_policy_gym_observation_carries_imu_features_post_step() -> None:
    """imu-proxy observations should populate angular-velocity / linear-acceleration features."""
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            action_type="twist",
            segment_duration_seconds=2.0,
        ),
    )
    observation, _ = adapter.reset(goal=unit_pose((0.5, 0.0, 0.0)))

    # Post-reset the kinematic state is the zero state — features are present
    # but step_dt_seconds == 0 signals the readings are not yet meaningful.
    expected_keys = {
        "imu-step-dt-seconds",
        "imu-angular-velocity-x",
        "imu-angular-velocity-y",
        "imu-angular-velocity-z",
        "imu-linear-acceleration-x",
        "imu-linear-acceleration-y",
        "imu-linear-acceleration-z",
    }
    assert expected_keys.issubset(observation.keys())
    assert observation["imu-step-dt-seconds"] == pytest.approx(0.0)
    assert observation["imu-angular-velocity-x"] == pytest.approx(0.0)
    assert observation["imu-linear-acceleration-x"] == pytest.approx(0.0)

    # A twist step over 2 s with a 0.25 m hop along +x produces a
    # finite-difference linear velocity of 0.125 m/s and (since the reset
    # state's velocity was zero) the same linear acceleration / 2 s = 0.0625
    # m/s^2 in the body frame on the +x axis.
    observation, _, _, _, _ = adapter.step({"target": {"x": 0.25, "y": 0.0, "z": 0.0}})
    assert observation["imu-step-dt-seconds"] == pytest.approx(2.0)
    assert observation["imu-linear-acceleration-x"] == pytest.approx(0.0625, abs=1e-6)
    assert observation["imu-linear-acceleration-y"] == pytest.approx(0.0, abs=1e-6)
    assert observation["imu-linear-acceleration-z"] == pytest.approx(0.0, abs=1e-6)
    # The agent does not rotate so body-frame angular velocity stays at zero.
    assert observation["imu-angular-velocity-x"] == pytest.approx(0.0, abs=1e-6)
    assert observation["imu-angular-velocity-y"] == pytest.approx(0.0, abs=1e-6)
    assert observation["imu-angular-velocity-z"] == pytest.approx(0.0, abs=1e-6)


def test_route_policy_gym_observation_drops_imu_features_when_render_unsupported() -> None:
    """Adapters wrapping an env without render_observation must omit imu-* features."""

    class StubEnv:
        """Minimal env that satisfies reset / sample_goal but not render_observation."""

        def __init__(self) -> None:
            self.dynamic_obstacles = None

        def reset(self, scene_id: str, *, seed: int | None = None) -> dict:
            pose = unit_pose((0.0, 0.0, 0.0))
            return {
                "scene": {"sceneId": scene_id},
                "state": {"sceneId": scene_id, "pose": pose.to_dict(), "stepIndex": 0},
                "backend": "stub",
            }

        def sample_goal(self, scene_id: str, *, seed: int | None = None):
            return unit_pose((0.5, 0.0, 0.0))

        def step(self, action) -> dict:  # pragma: no cover - not exercised below
            raise NotImplementedError

    adapter = RoutePolicyGymAdapter(StubEnv(), RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4))
    observation, _ = adapter.reset(goal=unit_pose((0.5, 0.0, 0.0)))

    assert not any(key.startswith("imu-") for key in observation)


def build_unit_catalog():
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


def unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
