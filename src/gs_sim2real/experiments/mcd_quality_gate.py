"""Quality-gate evaluation for collected MCD run reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MCDQualityGatePolicy:
    """Thresholds for deciding whether collected MCD quality runs are usable."""

    require_complete_artifacts: bool = True
    min_frame_fraction: float = 0.95
    min_depth_fraction: float = 0.95
    min_registered_fraction: float = 0.90
    min_sparse_points: int = 1
    min_trained_gaussians: int = 1
    min_splat_gaussians: int = 1
    require_final_l1: bool = True
    max_final_l1: float | None = None


def evaluate_mcd_quality_gates(
    report: dict[str, Any],
    policy: MCDQualityGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate all runs in a collected quality report against a gate policy."""
    active_policy = policy or MCDQualityGatePolicy()
    run_gates = [_evaluate_run_gate(run, active_policy) for run in report["runs"]]
    passed_count = sum(1 for run in run_gates if run["passed"])
    return {
        "type": "mcd-quality-gate-report",
        "sourceType": report.get("type"),
        "policy": asdict(active_policy),
        "runCount": len(run_gates),
        "passedCount": passed_count,
        "failedCount": len(run_gates) - passed_count,
        "passed": bool(run_gates) and passed_count == len(run_gates),
        "runs": run_gates,
    }


def render_quality_gate_markdown(gate_report: dict[str, Any]) -> str:
    """Render an evaluated MCD quality gate as a compact Markdown table."""
    lines = [
        "# MCD Quality Gate",
        "",
        f"Gate: {gate_report['passedCount']}/{gate_report['runCount']} runs passed",
        "",
        _render_policy_summary(gate_report["policy"]),
        "",
        "| Run | Gate | Failed checks |",
        "| --- | --- | --- |",
    ]
    for run in gate_report["runs"]:
        failed = [check["name"] for check in run["checks"] if not check["passed"]]
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    "pass" if run["passed"] else "fail",
                    ", ".join(failed) or "none",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _evaluate_run_gate(run: dict[str, Any], policy: MCDQualityGatePolicy) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    preprocess = run["preprocess"]
    train = run["train"]
    export = run["export"]
    planned_frames = run.get("plannedFrames")

    if policy.require_complete_artifacts:
        _append_check(
            checks,
            "artifacts",
            run.get("complete") is True,
            observed="complete" if run.get("complete") else "missing",
            expected="complete",
        )

    _append_fraction_check(
        checks,
        "frames",
        preprocess.get("imageCount"),
        planned_frames,
        policy.min_frame_fraction,
    )
    _append_fraction_check(
        checks,
        "depth",
        preprocess.get("depthMapCount"),
        preprocess.get("imageCount") or planned_frames,
        policy.min_depth_fraction,
    )
    registered_expected = preprocess.get("imageCount") or planned_frames
    _append_fraction_check(
        checks,
        "registered",
        preprocess.get("registeredImageCount"),
        registered_expected,
        policy.min_registered_fraction,
    )
    _append_min_check(checks, "sparse_points", preprocess.get("points3DCount"), policy.min_sparse_points)
    _append_min_check(checks, "trained_gaussians", train.get("trainedGaussians"), policy.min_trained_gaussians)
    _append_min_check(checks, "splat_gaussians", export.get("splatGaussians"), policy.min_splat_gaussians)
    if policy.require_final_l1:
        _append_check(
            checks,
            "final_l1",
            train.get("finalL1") is not None,
            observed=train.get("finalL1"),
            expected="present",
        )
    if policy.max_final_l1 is not None:
        value = train.get("finalL1")
        _append_check(
            checks,
            "final_l1_max",
            value is not None and float(value) <= policy.max_final_l1,
            observed=value,
            expected=f"<= {policy.max_final_l1:g}",
        )

    return {
        "name": run["name"],
        "label": run["label"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _append_fraction_check(
    checks: list[dict[str, Any]],
    name: str,
    actual: Any,
    expected_total: Any,
    minimum_fraction: float,
) -> None:
    if expected_total is None or int(expected_total) <= 0:
        _append_check(checks, name, False, observed=actual, expected="planned count")
        return
    if actual is None:
        _append_check(
            checks,
            name,
            False,
            observed=actual,
            expected=f">= {_format_percent(minimum_fraction)} of {int(expected_total)}",
        )
        return
    fraction = float(actual) / float(expected_total)
    _append_check(
        checks,
        name,
        fraction >= minimum_fraction,
        observed=f"{int(actual)} ({_format_percent(fraction)})",
        expected=f">= {_format_percent(minimum_fraction)} of {int(expected_total)}",
    )


def _append_min_check(checks: list[dict[str, Any]], name: str, actual: Any, minimum: int) -> None:
    _append_check(
        checks,
        name,
        actual is not None and int(actual) >= minimum,
        observed=actual,
        expected=f">= {minimum}",
    )


def _append_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def _render_policy_summary(policy: dict[str, Any]) -> str:
    parts = [
        f"frames >= {_format_percent(policy['min_frame_fraction'])}",
        f"depth >= {_format_percent(policy['min_depth_fraction'])}",
        f"registered >= {_format_percent(policy['min_registered_fraction'])}",
        f"sparse points >= {policy['min_sparse_points']}",
        f"trained gaussians >= {policy['min_trained_gaussians']}",
        f"splat gaussians >= {policy['min_splat_gaussians']}",
    ]
    if policy["require_complete_artifacts"]:
        parts.insert(0, "artifacts complete")
    if policy["require_final_l1"]:
        parts.append("final L1 present")
    if policy["max_final_l1"] is not None:
        parts.append(f"final L1 <= {policy['max_final_l1']:g}")
    return "Policy: " + ", ".join(parts)


def _format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


__all__ = [
    "MCDQualityGatePolicy",
    "evaluate_mcd_quality_gates",
    "render_quality_gate_markdown",
]
