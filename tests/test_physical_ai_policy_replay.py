"""Tests for route policy replay loading and offline training batches."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RouteRewardWeights,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    build_simulation_catalog,
    collect_route_policy_dataset,
    iter_route_policy_replay_batches,
    load_route_policy_dataset_json,
    load_route_policy_transitions_jsonl,
    route_policy_transition_table_from_rows,
    write_route_policy_dataset_json,
    write_route_policy_transitions_jsonl,
)


def test_load_route_policy_dataset_json_round_trips_episode_payload(tmp_path: Path) -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-replay",
        seed_start=5,
        goals=(unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0))),
        metadata={"split": "train"},
    )
    path = write_route_policy_dataset_json(tmp_path / "dataset.json", dataset)

    loaded = load_route_policy_dataset_json(path)

    assert loaded.to_dict() == dataset.to_dict()
    assert loaded.dataset_id == "unit-replay"
    assert loaded.transition_count == 2
    assert loaded.episodes[0].transitions[0].info["termination_reason"] == "goal-reached"


def test_load_route_policy_transitions_jsonl_builds_vector_batch(tmp_path: Path) -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-jsonl",
        goals=(unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0))),
    )
    path = write_route_policy_transitions_jsonl(tmp_path / "transitions.jsonl", dataset)

    table = load_route_policy_transitions_jsonl(path)
    schema = build_route_policy_replay_schema(table)
    batch = build_route_policy_replay_batch(table, schema=schema, metadata={"purpose": "imitation"})

    assert table.dataset_id == "unit-jsonl"
    assert table.transition_count == 2
    assert "goal-distance-meters" in schema.observation_keys
    assert "goal-reached" in schema.next_observation_keys
    assert "payload.target.position.0" in schema.action_keys

    target_x_index = schema.action_keys.index("payload.target.position.0")
    assert batch.size == 2
    assert batch.action_matrix[0][target_x_index] == pytest.approx(0.25)
    assert batch.reward_vector[0] > 0.0
    assert batch.done_vector == (True, True)
    assert batch.to_dict()["metadata"]["purpose"] == "imitation"


def test_route_policy_replay_schema_can_pin_training_columns(tmp_path: Path) -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=1,
        dataset_id="unit-pinned",
        goals=(unit_pose((0.25, 0.0, 0.0)),),
    )
    table = load_route_policy_transitions_jsonl(write_route_policy_transitions_jsonl(tmp_path / "rows.jsonl", dataset))

    schema = build_route_policy_replay_schema(
        table,
        observation_keys=("goal-distance-meters",),
        action_keys=("payload.target.position.0", "payload.target.position.9"),
        next_observation_keys=("goal-reached",),
    )
    batch = build_route_policy_replay_batch(table, schema=schema)

    assert schema.observation_keys == ("goal-distance-meters",)
    assert schema.action_keys == ("payload.target.position.0", "payload.target.position.9")
    assert batch.observation_matrix == (
        (pytest.approx(dataset.episodes[0].transitions[0].observation["goal-distance-meters"]),),
    )
    assert batch.action_matrix == ((0.25, 0.0),)
    assert batch.next_observation_matrix == ((1.0,),)


def test_iter_route_policy_replay_batches_uses_one_schema_and_deterministic_shuffle() -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        partial_goal_policy,
        episode_count=3,
        dataset_id="unit-batches",
        goals=(unit_pose((0.75, 0.0, 0.0)),),
        max_steps=1,
    )

    batches = tuple(iter_route_policy_replay_batches(dataset, batch_size=2))
    dropped = tuple(iter_route_policy_replay_batches(dataset, batch_size=2, drop_remainder=True))
    shuffled_a = tuple(iter_route_policy_replay_batches(dataset, batch_size=2, shuffle=True, seed=11))
    shuffled_b = tuple(iter_route_policy_replay_batches(dataset, batch_size=2, shuffle=True, seed=11))

    assert [batch.size for batch in batches] == [2, 1]
    assert [batch.size for batch in dropped] == [2]
    assert batches[0].schema == batches[1].schema
    assert [sample.episode_id for batch in shuffled_a for sample in batch.samples] == [
        sample.episode_id for batch in shuffled_b for sample in batch.samples
    ]


def test_transition_table_requires_dataset_id_for_mixed_jsonl_rows() -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-mixed",
        goals=(unit_pose((0.25, 0.0, 0.0)),),
    )
    rows = [dict(row) for row in dataset.transition_rows()]
    rows[1]["datasetId"] = "other-dataset"

    with pytest.raises(ValueError, match="dataset_id is required"):
        route_policy_transition_table_from_rows(rows)

    table = route_policy_transition_table_from_rows(rows, dataset_id="merged")

    assert table.dataset_id == "merged"
    assert table.metadata["sourceDatasetIds"] == ["other-dataset", "unit-mixed"]


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
