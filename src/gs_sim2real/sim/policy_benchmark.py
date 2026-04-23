"""Benchmark runner and report artifacts for route policy baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .contract import SimulationCatalog, load_simulation_catalog_from_scene_picker
from .gym_adapter import RoutePolicyGymAdapter, RoutePolicyEnvConfig
from .headless import HeadlessPhysicalAIEnvironment
from .interfaces import Pose3D
from .policy_dataset import RoutePolicyCallable, RoutePolicyGoal
from .policy_imitation import (
    RoutePolicyImitationFitConfig,
    RoutePolicyImitationModel,
    fit_route_policy_imitation_model,
    load_route_policy_imitation_model_json,
    write_route_policy_imitation_model_json,
)
from .policy_quality import (
    RoutePolicyBaselineEvaluation,
    RoutePolicyQualityThresholds,
    evaluate_route_policy_baselines,
)
from .policy_replay import (
    RoutePolicyReplaySource,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    load_route_policy_dataset_json,
    load_route_policy_transitions_jsonl,
)


ROUTE_POLICY_BENCHMARK_VERSION = "gs-mapper-route-policy-benchmark/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkReport:
    """JSON-friendly benchmark report for one route policy comparison run."""

    benchmark_id: str
    evaluation: RoutePolicyBaselineEvaluation
    model_summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_BENCHMARK_VERSION

    @property
    def best_policy_name(self) -> str | None:
        return self.evaluation.best_policy_name

    @property
    def passed(self) -> bool:
        return bool(self.evaluation.results) and all(result.passed for result in self.evaluation.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-benchmark-report",
            "version": self.version,
            "benchmarkId": self.benchmark_id,
            "passed": self.passed,
            "bestPolicyName": self.best_policy_name,
            "summary": _evaluation_summary(self.evaluation),
            "modelSummary": _json_mapping(self.model_summary),
            "evaluation": self.evaluation.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


def run_route_policy_imitation_benchmark(
    adapters: Sequence[RoutePolicyGymAdapter],
    model: RoutePolicyImitationModel,
    *,
    episode_count: int,
    benchmark_id: str = "route-policy-benchmark",
    policy_name: str = "imitation",
    include_direct_baseline: bool = False,
    seed_start: int = 0,
    goals: Sequence[RoutePolicyGoal] | None = None,
    max_steps: int | None = None,
    thresholds: RoutePolicyQualityThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyBenchmarkReport:
    """Evaluate a fitted imitation model, optionally beside a direct-goal baseline."""

    policies: dict[str, RoutePolicyCallable] = {}
    if include_direct_baseline:
        policies["direct"] = _direct_goal_policy
    policies[policy_name] = model.as_policy()
    evaluation = evaluate_route_policy_baselines(
        adapters,
        policies,
        episode_count=episode_count,
        evaluation_id=benchmark_id,
        seed_start=seed_start,
        goals=goals,
        max_steps=max_steps,
        thresholds=thresholds,
        metadata={
            "benchmarkId": benchmark_id,
            "policyName": policy_name,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
    )
    return RoutePolicyBenchmarkReport(
        benchmark_id=benchmark_id,
        evaluation=evaluation,
        model_summary=_model_summary(model),
        metadata={
            "policyName": policy_name,
            "includeDirectBaseline": include_direct_baseline,
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_benchmark_report_json(path: str | Path, report: RoutePolicyBenchmarkReport) -> Path:
    """Write a route policy benchmark report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def render_route_policy_benchmark_markdown(report: RoutePolicyBenchmarkReport) -> str:
    """Render a compact Markdown summary for CLI logs or CI artifacts."""

    lines = [
        f"# Route Policy Benchmark: {report.benchmark_id}",
        f"- Status: {'PASS' if report.passed else 'FAIL'}",
        f"- Best policy: {report.best_policy_name or 'n/a'}",
        f"- Policies: {len(report.evaluation.results)}",
        "",
        "| Policy | Pass | Success | Collision | Truncation | Mean reward |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report.evaluation.results:
        metrics = result.quality.metrics
        lines.append(
            "| "
            f"{result.policy_name} | "
            f"{'yes' if result.passed else 'no'} | "
            f"{_percent(metrics.get('success-rate', 0.0))} | "
            f"{_format_float(metrics.get('collision-rate', 0.0))} | "
            f"{_percent(metrics.get('truncation-rate', 0.0))} | "
            f"{_format_float(metrics.get('mean-reward', 0.0))} |"
        )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy benchmark CLI."""

    model, model_source = _load_or_fit_model(args)
    if getattr(args, "model_output", None):
        write_route_policy_imitation_model_json(args.model_output, model)

    catalog = load_simulation_catalog_from_scene_picker(
        getattr(args, "scene_catalog"),
        site_url=getattr(args, "site_url", "https://rsasaki0109.github.io/gs-mapper/"),
    )
    scene_id = _resolve_scene_id(catalog, getattr(args, "scene_id", None))
    adapter = RoutePolicyGymAdapter(
        HeadlessPhysicalAIEnvironment(catalog),
        RoutePolicyEnvConfig(
            scene_id=scene_id,
            max_steps=getattr(args, "max_steps", None) or RoutePolicyEnvConfig().max_steps,
        ),
    )
    report = run_route_policy_imitation_benchmark(
        (adapter,),
        model,
        episode_count=int(getattr(args, "episode_count")),
        benchmark_id=str(getattr(args, "benchmark_id")),
        policy_name=str(getattr(args, "policy_name")),
        include_direct_baseline=bool(getattr(args, "include_direct_baseline", False)),
        seed_start=int(getattr(args, "seed_start")),
        goals=_goals_from_args(catalog, scene_id, getattr(args, "goal", None)),
        max_steps=getattr(args, "max_steps", None),
        thresholds=_thresholds_from_args(args),
        metadata={
            "sceneCatalog": str(getattr(args, "scene_catalog")),
            "sceneId": scene_id,
            "modelSource": model_source,
        },
    )
    write_route_policy_benchmark_report_json(getattr(args, "output"), report)
    markdown = render_route_policy_benchmark_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Report saved to: {getattr(args, 'output')}")
    if getattr(args, "model_output", None):
        print(f"Model saved to: {getattr(args, 'model_output')}")


def _load_or_fit_model(args: Any) -> tuple[RoutePolicyImitationModel, str]:
    model_path = getattr(args, "model", None)
    dataset_json = getattr(args, "dataset_json", None)
    transitions_jsonl = getattr(args, "transitions_jsonl", None)
    training_sources = tuple(item for item in (dataset_json, transitions_jsonl) if item)
    if model_path and training_sources:
        raise ValueError("--model cannot be combined with --dataset-json or --transitions-jsonl")
    if model_path:
        return load_route_policy_imitation_model_json(model_path), str(model_path)
    if len(training_sources) != 1:
        raise ValueError("provide exactly one of --model, --dataset-json, or --transitions-jsonl")

    if dataset_json:
        source: RoutePolicyReplaySource = load_route_policy_dataset_json(dataset_json)
        source_label = str(dataset_json)
    else:
        assert transitions_jsonl is not None
        source = load_route_policy_transitions_jsonl(transitions_jsonl)
        source_label = str(transitions_jsonl)

    action_keys = getattr(args, "action_keys", None)
    schema = (
        build_route_policy_replay_schema(source, action_keys=tuple(action_keys))
        if action_keys is not None
        else build_route_policy_replay_schema(source)
    )
    batch = build_route_policy_replay_batch(source, schema=schema)
    model = fit_route_policy_imitation_model(
        batch,
        config=RoutePolicyImitationFitConfig(neighbor_count=int(getattr(args, "neighbor_count"))),
        metadata={"source": source_label},
    )
    return model, source_label


def _resolve_scene_id(catalog: SimulationCatalog, scene_id: str | None) -> str:
    if scene_id:
        catalog.scene_by_id(scene_id)
        return str(scene_id)
    if "outdoor-demo" in catalog.scene_ids():
        return "outdoor-demo"
    scene_ids = catalog.scene_ids()
    if not scene_ids:
        raise ValueError("simulation catalog must contain at least one scene")
    return scene_ids[0]


def _goals_from_args(
    catalog: SimulationCatalog,
    scene_id: str,
    goal_rows: Sequence[Sequence[float]] | None,
) -> tuple[Pose3D, ...] | None:
    if not goal_rows:
        return None
    frame_id = catalog.scene_by_id(scene_id).coordinate_frame.frame_id
    return tuple(
        Pose3D(
            position=_position_tuple(row),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=frame_id,
        )
        for row in goal_rows
    )


def _thresholds_from_args(args: Any) -> RoutePolicyQualityThresholds | None:
    values = {
        "min_success_rate": getattr(args, "min_success_rate", None),
        "max_collision_rate": getattr(args, "max_collision_rate", None),
        "max_truncation_rate": getattr(args, "max_truncation_rate", None),
        "min_episode_count": getattr(args, "min_episode_count", None),
        "min_transition_count": getattr(args, "min_transition_count", None),
    }
    if all(value is None for value in values.values()):
        return None
    defaults = RoutePolicyQualityThresholds()
    return RoutePolicyQualityThresholds(
        min_success_rate=defaults.min_success_rate
        if values["min_success_rate"] is None
        else float(values["min_success_rate"]),
        max_collision_rate=defaults.max_collision_rate
        if values["max_collision_rate"] is None
        else float(values["max_collision_rate"]),
        max_truncation_rate=defaults.max_truncation_rate
        if values["max_truncation_rate"] is None
        else float(values["max_truncation_rate"]),
        min_scene_count=defaults.min_scene_count,
        min_episode_count=defaults.min_episode_count
        if values["min_episode_count"] is None
        else int(values["min_episode_count"]),
        min_transition_count=defaults.min_transition_count
        if values["min_transition_count"] is None
        else int(values["min_transition_count"]),
    )


def _direct_goal_policy(observation: Mapping[str, float], info: Mapping[str, Any]) -> dict[str, Any]:
    del observation
    return {
        "routeId": f"direct-{info.get('episodeIndex', 0)}-{info.get('stepIndex', 0)}",
        "target": info["goal"],
    }


def _model_summary(model: RoutePolicyImitationModel) -> dict[str, Any]:
    return {
        "version": model.version,
        "sampleCount": model.sample_count,
        "schema": model.schema.to_dict(),
        "config": model.config.to_dict(),
        "metadata": _json_mapping(model.metadata),
    }


def _evaluation_summary(evaluation: RoutePolicyBaselineEvaluation) -> dict[str, Any]:
    return {
        "evaluationId": evaluation.evaluation_id,
        "bestPolicyName": evaluation.best_policy_name,
        "policyCount": len(evaluation.results),
        "policies": [
            {
                "policyName": result.policy_name,
                "passed": result.passed,
                "metrics": _float_mapping(result.quality.metrics),
                "failedChecks": list(result.quality.failed_checks),
            }
            for result in evaluation.results
        ],
    }


def _position_tuple(row: Sequence[float]) -> tuple[float, float, float]:
    if len(row) != 3:
        raise ValueError("goal must contain exactly three coordinates")
    return (float(row[0]), float(row[1]), float(row[2]))


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _float_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


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
