"""Tests for route policy rollout dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    ROUTE_POLICY_DATASET_VERSION,
    RoutePolicyDatasetExport,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RouteRewardWeights,
    build_simulation_catalog,
    collect_route_policy_dataset,
    collect_route_policy_episode,
    serialize_route_policy_action,
    write_route_policy_dataset_json,
    write_route_policy_transitions_jsonl,
)


def test_collect_route_policy_episode_records_replay_transition() -> None:
    adapter = build_adapter()

    episode = collect_route_policy_episode(
        adapter,
        direct_goal_policy,
        seed=3,
        goal=unit_pose((0.25, 0.0, 0.0)),
        episode_id="episode-direct",
    )

    assert episode.episode_id == "episode-direct"
    assert episode.scene_id == "unit-scene"
    assert episode.seed == 3
    assert episode.step_count == 1
    assert episode.terminated is True
    assert episode.truncated is False
    assert episode.summary()["goalReached"] is True
    assert episode.total_reward == pytest.approx(episode.transitions[0].reward)
    transition = episode.transitions[0]
    assert transition.step_index == 0
    assert transition.action["kind"] == "mapping"
    assert transition.action["payload"]["routeId"] == "direct-0"
    assert transition.next_observation["goal-reached"] == 1.0
    assert transition.info["policySample"]["observation"]["sourceType"] == "route-rollout"
    assert transition.to_dict()["recordType"] == "route-policy-transition"


def test_collect_route_policy_episode_marks_collector_step_limit_as_truncated() -> None:
    adapter = build_adapter(max_steps=8)

    episode = collect_route_policy_episode(
        adapter,
        partial_goal_policy,
        goal=unit_pose((0.75, 0.0, 0.0)),
        max_steps=1,
    )

    assert episode.step_count == 1
    assert episode.terminated is False
    assert episode.truncated is True
    assert episode.summary()["terminationReason"] == "collector-max-steps"
    assert episode.transitions[0].info["done"] is True


def test_collect_route_policy_dataset_round_robins_adapters_and_exports_json(tmp_path: Path) -> None:
    adapters = (build_adapter(), build_adapter())

    dataset = collect_route_policy_dataset(
        adapters,
        direct_goal_policy,
        episode_count=3,
        dataset_id="unit-rollouts",
        seed_start=10,
        goals=(unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0))),
        metadata={"split": "train"},
    )

    assert dataset.version == ROUTE_POLICY_DATASET_VERSION
    assert dataset.dataset_id == "unit-rollouts"
    assert len(dataset.episodes) == 3
    assert dataset.transition_count == 3
    assert [episode.episode_id for episode in dataset.episodes] == [
        "unit-rollouts-episode-000000",
        "unit-rollouts-episode-000001",
        "unit-rollouts-episode-000002",
    ]
    assert [episode.seed for episode in dataset.episodes] == [10, 11, 12]
    assert [episode.metadata["adapterIndex"] for episode in dataset.episodes] == [0, 1, 0]

    json_path = write_route_policy_dataset_json(tmp_path / "dataset.json", dataset)
    jsonl_path = write_route_policy_transitions_jsonl(tmp_path / "transitions.jsonl", dataset)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert payload["recordType"] == "route-policy-dataset"
    assert payload["metadata"]["split"] == "train"
    assert payload["episodeCount"] == 3
    assert payload["transitionCount"] == 3
    assert len(rows) == 3
    assert all(row["datasetId"] == "unit-rollouts" for row in rows)
    assert rows[0]["recordType"] == "route-policy-transition"


def test_route_policy_dataset_export_reports_transition_rows() -> None:
    episode = collect_route_policy_episode(
        build_adapter(),
        direct_goal_policy,
        goal=unit_pose((0.25, 0.0, 0.0)),
        episode_id="manual-episode",
    )
    dataset = RoutePolicyDatasetExport(dataset_id="manual", episodes=(episode,))

    (row,) = dataset.transition_rows()

    assert dataset.transition_count == 1
    assert row["datasetId"] == "manual"
    assert row["episodeId"] == "manual-episode"
    assert dataset.to_dict()["episodes"][0]["summary"]["goalReached"] is True


def test_serialize_route_policy_action_keeps_position_actions_json_friendly() -> None:
    action = serialize_route_policy_action((0.1, 0.2, 0.3))

    assert action == {"kind": "position", "payload": [0.1, 0.2, 0.3]}


def build_adapter(*, max_steps: int = 4) -> RoutePolicyGymAdapter:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    return RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            max_steps=max_steps,
            goal_reward=2.0,
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
        ),
    )


def direct_goal_policy(observation, info):
    del observation
    return {
        "routeId": f"direct-{info['stepIndex']}",
        "target": info["goal"],
    }


def partial_goal_policy(observation, info):
    del observation
    pose = info["pose"]["position"]
    goal = info["goal"]["position"]
    return {
        "routeId": f"partial-{info['stepIndex']}",
        "target": {
            "x": pose[0] + (goal[0] - pose[0]) * 0.25,
            "y": pose[1] + (goal[1] - pose[1]) * 0.25,
            "z": pose[2] + (goal[2] - pose[2]) * 0.25,
        },
    }


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
