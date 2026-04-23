"""Tests for route policy benchmark reports and CLI runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs_sim2real import cli
from gs_sim2real.cli import build_parser
from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyBenchmarkRegressionThresholds,
    RoutePolicyGoalSpec,
    RoutePolicyGoalSuite,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyRegistry,
    RoutePolicyRegistryEntry,
    RoutePolicyScenarioSet,
    RoutePolicyScenarioSpec,
    RouteRewardWeights,
    build_route_policy_benchmark_history,
    build_simulation_catalog,
    collect_route_policy_dataset,
    load_route_policy_benchmark_history_json,
    load_route_policy_goal_suite_json,
    load_route_policy_imitation_model_json,
    load_route_policy_registry_json,
    load_route_policy_scenario_set_run_json,
    render_route_policy_benchmark_history_markdown,
    render_route_policy_benchmark_markdown,
    render_route_policy_scenario_set_markdown,
    run_route_policy_imitation_benchmark,
    run_route_policy_registry_benchmark,
    run_route_policy_scenario_set,
    write_route_policy_benchmark_history_json,
    write_route_policy_goal_suite_json,
    write_route_policy_imitation_model_json,
    write_route_policy_registry_json,
    write_route_policy_scenario_set_json,
    write_route_policy_scenario_set_run_json,
    write_route_policy_transitions_jsonl,
)


def test_run_route_policy_imitation_benchmark_compares_direct_baseline() -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-benchmark",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))

    report = run_route_policy_imitation_benchmark(
        (build_adapter(),),
        model,
        episode_count=2,
        benchmark_id="unit-benchmark",
        include_direct_baseline=True,
        goals=goals,
        max_steps=4,
    )

    assert report.passed
    assert report.best_policy_name in {"direct", "imitation"}
    assert [result.policy_name for result in report.evaluation.results] == ["direct", "imitation"]
    assert "Route Policy Benchmark: unit-benchmark" in render_route_policy_benchmark_markdown(report)


def test_route_policy_benchmark_cli_fits_saves_and_writes_reports(tmp_path: Path) -> None:
    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-benchmark-cli",
        goals=goals,
    )
    transitions_path = write_route_policy_transitions_jsonl(tmp_path / "transitions.jsonl", dataset)
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    model_path = tmp_path / "model.json"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark",
            "--transitions-jsonl",
            str(transitions_path),
            "--scene-catalog",
            str(catalog_path),
            "--scene-id",
            "unit-scene",
            "--benchmark-id",
            "unit-cli-benchmark",
            "--episode-count",
            "2",
            "--seed-start",
            "0",
            "--max-steps",
            "4",
            "--goal",
            "0.25",
            "0.0",
            "0.0",
            "--goal",
            "0.5",
            "0.0",
            "0.0",
            "--action-keys",
            *target_position_keys(),
            "--include-direct-baseline",
            "--min-success-rate",
            "1.0",
            "--max-collision-rate",
            "0.0",
            "--max-truncation-rate",
            "0.0",
            "--min-episode-count",
            "2",
            "--min-transition-count",
            "2",
            "--model-output",
            str(model_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_benchmark(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    model = load_route_policy_imitation_model_json(model_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert report["recordType"] == "route-policy-benchmark-report"
    assert report["benchmarkId"] == "unit-cli-benchmark"
    assert report["passed"] is True
    assert report["summary"]["policyCount"] == 2
    assert report["modelSummary"]["sampleCount"] == 2
    assert model.sample_count == 2
    assert "| Policy | Pass | Success |" in markdown


def test_route_policy_registry_benchmark_loads_named_policies_and_goal_suite(tmp_path: Path) -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-registry-train",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))
    model_path = write_route_policy_imitation_model_json(tmp_path / "imitation.json", model)
    registry_path = write_route_policy_registry_json(
        tmp_path / "policies.json",
        RoutePolicyRegistry(
            registry_id="unit-policies",
            policies=(
                RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
                RoutePolicyRegistryEntry(
                    policy_name="imitation-k1",
                    policy_type="imitation-model",
                    model_path=model_path.name,
                ),
            ),
        ),
    )
    suite_path = write_route_policy_goal_suite_json(
        tmp_path / "goals.json",
        RoutePolicyGoalSuite(
            suite_id="unit-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(
                RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
                RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
            ),
        ),
    )

    registry = load_route_policy_registry_json(registry_path)
    suite = load_route_policy_goal_suite_json(suite_path)
    report = run_route_policy_registry_benchmark(
        (build_adapter(),),
        registry,
        episode_count=2,
        benchmark_id="unit-registry-benchmark",
        registry_base_path=tmp_path,
        goals=suite.to_goals(frame_id="generic_world"),
        max_steps=4,
    )

    assert report.passed
    assert [result.policy_name for result in report.evaluation.results] == ["direct", "imitation-k1"]
    assert report.to_dict()["modelSummary"]["registry"]["registryId"] == "unit-policies"
    assert report.to_dict()["modelSummary"]["policies"][1]["sampleCount"] == 2


def test_route_policy_benchmark_cli_uses_registry_and_goal_suite(tmp_path: Path) -> None:
    from gs_sim2real.sim import build_route_policy_replay_batch, build_route_policy_replay_schema
    from gs_sim2real.sim import fit_route_policy_imitation_model

    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-policy-registry-cli-train",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))
    write_route_policy_imitation_model_json(tmp_path / "model.json", model)
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-registry",
            policies=(
                RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
                RoutePolicyRegistryEntry(
                    policy_name="imitation",
                    policy_type="imitation-model",
                    model_path="model.json",
                ),
            ),
        ),
    )
    suite_path = write_route_policy_goal_suite_json(
        tmp_path / "goal-suite.json",
        RoutePolicyGoalSuite(
            suite_id="unit-cli-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(
                RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
                RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
            ),
        ),
    )
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    output_path = tmp_path / "registry-report.json"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark",
            "--policy-registry",
            str(registry_path),
            "--goal-suite",
            str(suite_path),
            "--scene-catalog",
            str(catalog_path),
            "--benchmark-id",
            "unit-cli-registry-benchmark",
            "--episode-count",
            "2",
            "--max-steps",
            "4",
            "--output",
            str(output_path),
        ]
    )

    cli.cmd_route_policy_benchmark(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["benchmarkId"] == "unit-cli-registry-benchmark"
    assert report["passed"] is True
    assert report["summary"]["policyCount"] == 2
    assert [policy["policyName"] for policy in report["summary"]["policies"]] == ["direct", "imitation"]
    assert report["modelSummary"]["registry"]["registryId"] == "unit-cli-registry"
    assert report["metadata"]["goalSuite"] == str(suite_path)


def test_route_policy_benchmark_history_flags_baseline_regressions(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline-report.json",
        benchmark_id="unit-history-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current-report.json",
        benchmark_id="unit-history-current",
        policies={
            "direct": {
                "success-rate": 0.9,
                "collision-rate": 0.03,
                "truncation-rate": 0.0,
                "mean-reward": 1.6,
            }
        },
    )

    history = build_route_policy_benchmark_history(
        (current_path,),
        baseline_report=baseline_path,
        history_id="unit-history",
        thresholds=RoutePolicyBenchmarkRegressionThresholds(
            max_success_rate_drop=0.05,
            max_collision_rate_increase=0.01,
            max_truncation_rate_increase=0.01,
            max_mean_reward_drop=0.25,
        ),
    )
    history_path = write_route_policy_benchmark_history_json(tmp_path / "history.json", history)
    loaded = load_route_policy_benchmark_history_json(history_path)
    failed_checks = set(loaded.failed_checks)

    assert not loaded.passed
    assert "success-rate-regression:current-report:direct" in failed_checks
    assert "collision-rate-regression:current-report:direct" in failed_checks
    assert "mean-reward-regression:current-report:direct" in failed_checks
    assert loaded.to_dict()["aggregate"][0]["metrics"]["success-rate"]["mean"] == 0.9
    assert "Regression Gate" in render_route_policy_benchmark_history_markdown(loaded)


def test_route_policy_benchmark_history_cli_writes_artifacts(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline.json",
        benchmark_id="unit-cli-history-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current.json",
        benchmark_id="unit-cli-history-current",
        policies={
            "direct": {
                "success-rate": 0.97,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 1.95,
            }
        },
    )
    output_path = tmp_path / "history.json"
    markdown_path = tmp_path / "history.md"

    args = build_parser().parse_args(
        [
            "route-policy-benchmark-history",
            "--report",
            str(current_path),
            "--baseline-report",
            str(baseline_path),
            "--history-id",
            "unit-cli-history",
            "--max-success-rate-drop",
            "0.05",
            "--max-collision-rate-increase",
            "0.01",
            "--max-truncation-rate-increase",
            "0.01",
            "--max-mean-reward-drop",
            "0.10",
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    cli.cmd_route_policy_benchmark_history(args)

    history = json.loads(output_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert history["recordType"] == "route-policy-benchmark-history"
    assert history["historyId"] == "unit-cli-history"
    assert history["passed"] is True
    assert history["reportCount"] == 1
    assert history["regressionChecks"][0]["checkId"] == "success-rate-regression:current:direct"
    assert "Route Policy Benchmark History: unit-cli-history" in markdown


def test_route_policy_benchmark_history_cli_exits_on_regression(tmp_path: Path) -> None:
    baseline_path = write_benchmark_report_fixture(
        tmp_path / "baseline.json",
        benchmark_id="unit-cli-gate-baseline",
        policies={
            "direct": {
                "success-rate": 1.0,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    current_path = write_benchmark_report_fixture(
        tmp_path / "current.json",
        benchmark_id="unit-cli-gate-current",
        policies={
            "direct": {
                "success-rate": 0.5,
                "collision-rate": 0.0,
                "truncation-rate": 0.0,
                "mean-reward": 2.0,
            }
        },
    )
    output_path = tmp_path / "history.json"
    args = build_parser().parse_args(
        [
            "route-policy-benchmark-history",
            "--report",
            str(current_path),
            "--baseline-report",
            str(baseline_path),
            "--max-success-rate-drop",
            "0.05",
            "--output",
            str(output_path),
            "--fail-on-regression",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_route_policy_benchmark_history(args)

    assert exc_info.value.code == 2
    history = json.loads(output_path.read_text(encoding="utf-8"))
    assert history["passed"] is False


def test_route_policy_scenario_set_runs_registry_across_goal_suites(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    registry_path = write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-scenario-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    near_goals = write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    far_goals = write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    scenario_set = RoutePolicyScenarioSet(
        scenario_set_id="unit-scenarios",
        policy_registry_path=registry_path.name,
        episode_count=1,
        seed_start=0,
        max_steps=4,
        scenarios=(
            RoutePolicyScenarioSpec(
                scenario_id="near",
                scene_catalog=catalog_path.name,
                goal_suite_path=near_goals.name,
            ),
            RoutePolicyScenarioSpec(
                scenario_id="far",
                scene_catalog=catalog_path.name,
                goal_suite_path=far_goals.name,
            ),
        ),
    )
    registry = load_route_policy_registry_json(registry_path)

    report = run_route_policy_scenario_set(
        scenario_set,
        registry,
        report_dir=tmp_path / "reports",
        scenario_set_base_path=tmp_path,
        registry_base_path=tmp_path,
        policy_registry_path=registry_path,
        history_output=tmp_path / "history.json",
        history_markdown_output=tmp_path / "history.md",
    )
    run_path = write_route_policy_scenario_set_run_json(tmp_path / "scenario-run.json", report)
    loaded = load_route_policy_scenario_set_run_json(run_path)

    assert loaded.passed
    assert loaded.scenario_count == 2
    assert loaded.history.to_dict()["reportCount"] == 2
    assert [result.scenario_id for result in loaded.scenario_results] == ["near", "far"]
    assert all(Path(result.report_path).exists() for result in loaded.scenario_results)
    assert (tmp_path / "history.json").exists()
    assert "Route Policy Scenario Set: unit-scenarios" in render_route_policy_scenario_set_markdown(loaded)


def test_route_policy_scenario_set_cli_writes_reports_and_history(tmp_path: Path) -> None:
    catalog_path = write_unit_scene_catalog(tmp_path / "scenes.json")
    write_route_policy_registry_json(
        tmp_path / "registry.json",
        RoutePolicyRegistry(
            registry_id="unit-cli-scenario-registry",
            policies=(RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "near-goals.json",
        RoutePolicyGoalSuite(
            suite_id="near-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),),
        ),
    )
    write_route_policy_goal_suite_json(
        tmp_path / "far-goals.json",
        RoutePolicyGoalSuite(
            suite_id="far-goals",
            scene_id="unit-scene",
            frame_id="generic_world",
            goals=(RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),),
        ),
    )
    scenario_set_path = write_route_policy_scenario_set_json(
        tmp_path / "scenarios.json",
        RoutePolicyScenarioSet(
            scenario_set_id="unit-cli-scenarios",
            policy_registry_path="registry.json",
            episode_count=1,
            seed_start=0,
            max_steps=4,
            scenarios=(
                RoutePolicyScenarioSpec(
                    scenario_id="near",
                    scene_catalog=catalog_path.name,
                    goal_suite_path="near-goals.json",
                ),
                RoutePolicyScenarioSpec(
                    scenario_id="far",
                    scene_catalog=catalog_path.name,
                    goal_suite_path="far-goals.json",
                ),
            ),
        ),
    )
    output_path = tmp_path / "scenario-run.json"
    markdown_path = tmp_path / "scenario-run.md"
    history_path = tmp_path / "history.json"
    history_markdown_path = tmp_path / "history.md"

    args = build_parser().parse_args(
        [
            "route-policy-scenario-set",
            "--scenario-set",
            str(scenario_set_path),
            "--report-dir",
            str(tmp_path / "reports"),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--history-output",
            str(history_path),
            "--history-markdown-output",
            str(history_markdown_path),
        ]
    )

    cli.cmd_route_policy_scenario_set(args)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert report["recordType"] == "route-policy-scenario-set-run"
    assert report["passed"] is True
    assert report["scenarioSetId"] == "unit-cli-scenarios"
    assert [result["scenarioId"] for result in report["scenarioResults"]] == ["near", "far"]
    assert history["recordType"] == "route-policy-benchmark-history"
    assert history["reportCount"] == 2
    assert "Route Policy Scenario Set: unit-cli-scenarios" in markdown_path.read_text(encoding="utf-8")
    assert "Route Policy Benchmark History" in history_markdown_path.read_text(encoding="utf-8")


def target_position_keys() -> tuple[str, str, str]:
    return ("payload.target.position.0", "payload.target.position.1", "payload.target.position.2")


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


def write_unit_scene_catalog(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "url": "assets/unit-scene/unit-scene.splat",
                        "label": "Unit Scene",
                        "summary": "Generic unit scene",
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_benchmark_report_fixture(
    path: Path,
    *,
    benchmark_id: str,
    policies: dict[str, dict[str, float]],
    passed: bool = True,
) -> Path:
    policy_rows = [
        {
            "policyName": policy_name,
            "passed": passed,
            "metrics": metrics,
            "failedChecks": [],
        }
        for policy_name, metrics in policies.items()
    ]
    path.write_text(
        json.dumps(
            {
                "recordType": "route-policy-benchmark-report",
                "version": "gs-mapper-route-policy-benchmark/v1",
                "benchmarkId": benchmark_id,
                "passed": passed,
                "bestPolicyName": next(iter(policies)),
                "summary": {
                    "evaluationId": benchmark_id,
                    "bestPolicyName": next(iter(policies)),
                    "policyCount": len(policy_rows),
                    "policies": policy_rows,
                },
                "modelSummary": {},
                "metadata": {"sceneId": "unit-scene"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
