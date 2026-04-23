"""History aggregation and regression gates for route policy benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .policy_benchmark import ROUTE_POLICY_BENCHMARK_VERSION


ROUTE_POLICY_BENCHMARK_HISTORY_VERSION = "gs-mapper-route-policy-benchmark-history/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkPolicySnapshot:
    """Compact per-policy metrics extracted from one benchmark report."""

    policy_name: str
    passed: bool
    metrics: Mapping[str, float]
    failed_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.policy_name):
            raise ValueError("policy_name must not be empty")
        object.__setattr__(self, "metrics", _float_mapping(self.metrics))
        object.__setattr__(self, "failed_checks", tuple(str(check) for check in self.failed_checks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-benchmark-policy-snapshot",
            "policyName": self.policy_name,
            "passed": bool(self.passed),
            "metrics": _float_mapping(self.metrics),
            "failedChecks": list(self.failed_checks),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkSnapshot:
    """Compact report snapshot suitable for history files and CI comparisons."""

    benchmark_id: str
    passed: bool
    best_policy_name: str | None
    policies: tuple[RoutePolicyBenchmarkPolicySnapshot, ...]
    source_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_BENCHMARK_VERSION

    def __post_init__(self) -> None:
        if not str(self.benchmark_id):
            raise ValueError("benchmark_id must not be empty")
        policy_names = tuple(policy.policy_name for policy in self.policies)
        if len(set(policy_names)) != len(policy_names):
            raise ValueError("benchmark snapshot must not contain duplicate policy names")

    @property
    def policy_count(self) -> int:
        return len(self.policies)

    def policy_by_name(self) -> dict[str, RoutePolicyBenchmarkPolicySnapshot]:
        return {policy.policy_name: policy for policy in self.policies}

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-benchmark-snapshot",
            "version": self.version,
            "sourcePath": self.source_path,
            "benchmarkId": self.benchmark_id,
            "passed": bool(self.passed),
            "bestPolicyName": self.best_policy_name,
            "policyCount": self.policy_count,
            "policies": [policy.to_dict() for policy in self.policies],
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkRegressionThresholds:
    """Allowed metric movement when comparing a current report against a baseline."""

    max_success_rate_drop: float = 0.0
    max_collision_rate_increase: float = 0.0
    max_truncation_rate_increase: float = 0.0
    max_mean_reward_drop: float | None = None
    require_baseline_policies: bool = True
    fail_on_report_failure: bool = True

    def __post_init__(self) -> None:
        _non_negative_float(self.max_success_rate_drop, "max_success_rate_drop")
        _non_negative_float(self.max_collision_rate_increase, "max_collision_rate_increase")
        _non_negative_float(self.max_truncation_rate_increase, "max_truncation_rate_increase")
        if self.max_mean_reward_drop is not None:
            _non_negative_float(self.max_mean_reward_drop, "max_mean_reward_drop")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxSuccessRateDrop": float(self.max_success_rate_drop),
            "maxCollisionRateIncrease": float(self.max_collision_rate_increase),
            "maxTruncationRateIncrease": float(self.max_truncation_rate_increase),
            "maxMeanRewardDrop": None if self.max_mean_reward_drop is None else float(self.max_mean_reward_drop),
            "requireBaselinePolicies": bool(self.require_baseline_policies),
            "failOnReportFailure": bool(self.fail_on_report_failure),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkRegressionCheck:
    """One stable pass/fail check produced by the history regression gate."""

    check_id: str
    passed: bool
    policy_name: str | None
    metric_name: str | None
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    allowed_delta: float | None = None
    source_path: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not str(self.check_id):
            raise ValueError("check_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-benchmark-regression-check",
            "checkId": self.check_id,
            "passed": bool(self.passed),
            "policyName": self.policy_name,
            "metricName": self.metric_name,
            "baselineValue": self.baseline_value,
            "currentValue": self.current_value,
            "delta": self.delta,
            "allowedDelta": self.allowed_delta,
            "sourcePath": self.source_path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyBenchmarkHistoryReport:
    """Versioned benchmark history with optional regression gate checks."""

    history_id: str
    reports: tuple[RoutePolicyBenchmarkSnapshot, ...]
    baseline_report: RoutePolicyBenchmarkSnapshot | None = None
    thresholds: RoutePolicyBenchmarkRegressionThresholds = field(
        default_factory=RoutePolicyBenchmarkRegressionThresholds
    )
    regression_checks: tuple[RoutePolicyBenchmarkRegressionCheck, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_BENCHMARK_HISTORY_VERSION

    def __post_init__(self) -> None:
        if not str(self.history_id):
            raise ValueError("history_id must not be empty")
        if not self.reports:
            raise ValueError("history report must contain at least one benchmark report")

    @property
    def passed(self) -> bool:
        if self.thresholds.fail_on_report_failure:
            if self.baseline_report is not None and not self.baseline_report.passed:
                return False
            if not all(report.passed for report in self.reports):
                return False
        return all(check.passed for check in self.regression_checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.regression_checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        aggregate = _aggregate_policies(self.reports)
        return {
            "recordType": "route-policy-benchmark-history",
            "version": self.version,
            "historyId": self.history_id,
            "passed": self.passed,
            "failedChecks": list(self.failed_checks),
            "reportCount": len(self.reports),
            "baselineReport": None if self.baseline_report is None else self.baseline_report.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "reports": [report.to_dict() for report in self.reports],
            "aggregate": aggregate,
            "regressionChecks": [check.to_dict() for check in self.regression_checks],
            "metadata": _json_mapping(self.metadata),
        }


def build_route_policy_benchmark_history(
    report_paths: Sequence[str | Path],
    *,
    baseline_report: str | Path | RoutePolicyBenchmarkSnapshot | None = None,
    history_id: str = "route-policy-benchmark-history",
    thresholds: RoutePolicyBenchmarkRegressionThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyBenchmarkHistoryReport:
    """Load benchmark reports, aggregate trends, and optionally compare them to a baseline."""

    if not report_paths:
        raise ValueError("report_paths must contain at least one report")
    reports = tuple(load_route_policy_benchmark_snapshot_json(path) for path in report_paths)
    resolved_thresholds = thresholds or RoutePolicyBenchmarkRegressionThresholds()
    resolved_baseline = _resolve_baseline_snapshot(baseline_report)
    checks: list[RoutePolicyBenchmarkRegressionCheck] = []
    if resolved_thresholds.fail_on_report_failure:
        if resolved_baseline is not None and not resolved_baseline.passed:
            checks.append(
                RoutePolicyBenchmarkRegressionCheck(
                    check_id="baseline-report-failed",
                    passed=False,
                    policy_name=None,
                    metric_name=None,
                    source_path=resolved_baseline.source_path,
                    message=f"baseline report {resolved_baseline.benchmark_id} did not pass",
                )
            )
        for report in reports:
            if not report.passed:
                checks.append(
                    RoutePolicyBenchmarkRegressionCheck(
                        check_id=f"report-failed:{_snapshot_key(report)}",
                        passed=False,
                        policy_name=None,
                        metric_name=None,
                        source_path=report.source_path,
                        message=f"benchmark report {report.benchmark_id} did not pass",
                    )
                )
    if resolved_baseline is not None:
        for report in reports:
            checks.extend(
                compare_route_policy_benchmark_snapshots(
                    report,
                    resolved_baseline,
                    thresholds=resolved_thresholds,
                )
            )
    return RoutePolicyBenchmarkHistoryReport(
        history_id=history_id,
        reports=reports,
        baseline_report=resolved_baseline,
        thresholds=resolved_thresholds,
        regression_checks=tuple(checks),
        metadata=_json_mapping(metadata or {}),
    )


def compare_route_policy_benchmark_snapshots(
    current: RoutePolicyBenchmarkSnapshot,
    baseline: RoutePolicyBenchmarkSnapshot,
    *,
    thresholds: RoutePolicyBenchmarkRegressionThresholds | None = None,
) -> tuple[RoutePolicyBenchmarkRegressionCheck, ...]:
    """Compare one current benchmark snapshot against one blessed baseline snapshot."""

    resolved_thresholds = thresholds or RoutePolicyBenchmarkRegressionThresholds()
    current_policies = current.policy_by_name()
    baseline_policies = baseline.policy_by_name()
    policy_names = baseline_policies.keys()
    current_key = _snapshot_key(current)
    checks: list[RoutePolicyBenchmarkRegressionCheck] = []

    for policy_name in policy_names:
        baseline_policy = baseline_policies[policy_name]
        current_policy = current_policies.get(policy_name)
        if current_policy is None:
            if resolved_thresholds.require_baseline_policies:
                checks.append(
                    RoutePolicyBenchmarkRegressionCheck(
                        check_id=f"missing-policy:{current_key}:{_slug(policy_name)}",
                        passed=False,
                        policy_name=policy_name,
                        metric_name=None,
                        source_path=current.source_path,
                        message=f"current report is missing baseline policy {policy_name}",
                    )
                )
            continue
        checks.extend(
            _metric_regression_checks(
                current_policy,
                baseline_policy,
                current_source_path=current.source_path,
                current_key=current_key,
                thresholds=resolved_thresholds,
            )
        )
    return tuple(checks)


def write_route_policy_benchmark_history_json(
    path: str | Path,
    history: RoutePolicyBenchmarkHistoryReport,
) -> Path:
    """Write a benchmark history artifact as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(history.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_benchmark_history_json(path: str | Path) -> RoutePolicyBenchmarkHistoryReport:
    """Load a route policy benchmark history JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_benchmark_history_from_dict(_mapping(payload, "history"))


def load_route_policy_benchmark_snapshot_json(path: str | Path) -> RoutePolicyBenchmarkSnapshot:
    """Load a benchmark report JSON as a compact snapshot."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return route_policy_benchmark_snapshot_from_report_dict(
        _mapping(payload, "benchmarkReport"),
        source_path=source_path.as_posix(),
    )


def route_policy_benchmark_history_from_dict(payload: Mapping[str, Any]) -> RoutePolicyBenchmarkHistoryReport:
    """Rebuild a benchmark history report from its JSON payload."""

    _record_type(payload, "route-policy-benchmark-history")
    version = str(payload.get("version", ROUTE_POLICY_BENCHMARK_HISTORY_VERSION))
    if version != ROUTE_POLICY_BENCHMARK_HISTORY_VERSION:
        raise ValueError(f"unsupported route policy benchmark history version: {version}")
    baseline_payload = payload.get("baselineReport")
    thresholds = route_policy_benchmark_regression_thresholds_from_dict(
        _mapping(payload.get("thresholds", {}), "thresholds")
    )
    return RoutePolicyBenchmarkHistoryReport(
        history_id=str(payload["historyId"]),
        reports=tuple(
            route_policy_benchmark_snapshot_from_dict(_mapping(item, "report"))
            for item in _sequence(payload.get("reports", ()), "reports")
        ),
        baseline_report=None
        if baseline_payload is None
        else route_policy_benchmark_snapshot_from_dict(_mapping(baseline_payload, "baselineReport")),
        thresholds=thresholds,
        regression_checks=tuple(
            route_policy_benchmark_regression_check_from_dict(_mapping(item, "regressionCheck"))
            for item in _sequence(payload.get("regressionChecks", ()), "regressionChecks")
        ),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_benchmark_snapshot_from_report_dict(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
) -> RoutePolicyBenchmarkSnapshot:
    """Extract a compact benchmark snapshot from a full benchmark report payload."""

    _record_type(payload, "route-policy-benchmark-report")
    version = str(payload.get("version", ROUTE_POLICY_BENCHMARK_VERSION))
    if version != ROUTE_POLICY_BENCHMARK_VERSION:
        raise ValueError(f"unsupported route policy benchmark version: {version}")
    summary = _mapping(payload.get("summary", {}), "summary")
    policies = tuple(
        RoutePolicyBenchmarkPolicySnapshot(
            policy_name=str(policy["policyName"]),
            passed=bool(policy.get("passed", False)),
            metrics=_float_mapping(_mapping(policy.get("metrics", {}), "metrics")),
            failed_checks=tuple(str(check) for check in _sequence(policy.get("failedChecks", ()), "failedChecks")),
        )
        for policy in (_mapping(item, "policy") for item in _sequence(summary.get("policies", ()), "policies"))
    )
    expected_policy_count = summary.get("policyCount")
    if expected_policy_count is not None and int(expected_policy_count) != len(policies):
        raise ValueError("summary policyCount does not match loaded policies")
    return RoutePolicyBenchmarkSnapshot(
        benchmark_id=str(payload["benchmarkId"]),
        passed=bool(payload.get("passed", False)),
        best_policy_name=None if payload.get("bestPolicyName") is None else str(payload["bestPolicyName"]),
        policies=policies,
        source_path=source_path,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_benchmark_snapshot_from_dict(payload: Mapping[str, Any]) -> RoutePolicyBenchmarkSnapshot:
    """Rebuild a compact benchmark snapshot from history JSON."""

    _record_type(payload, "route-policy-benchmark-snapshot")
    version = str(payload.get("version", ROUTE_POLICY_BENCHMARK_VERSION))
    if version != ROUTE_POLICY_BENCHMARK_VERSION:
        raise ValueError(f"unsupported route policy benchmark version: {version}")
    policies = tuple(
        route_policy_benchmark_policy_snapshot_from_dict(_mapping(item, "policy"))
        for item in _sequence(payload.get("policies", ()), "policies")
    )
    expected_policy_count = payload.get("policyCount")
    if expected_policy_count is not None and int(expected_policy_count) != len(policies):
        raise ValueError("snapshot policyCount does not match loaded policies")
    return RoutePolicyBenchmarkSnapshot(
        benchmark_id=str(payload["benchmarkId"]),
        passed=bool(payload.get("passed", False)),
        best_policy_name=None if payload.get("bestPolicyName") is None else str(payload["bestPolicyName"]),
        policies=policies,
        source_path=None if payload.get("sourcePath") is None else str(payload["sourcePath"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_benchmark_policy_snapshot_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyBenchmarkPolicySnapshot:
    """Rebuild one policy snapshot from history JSON."""

    _record_type(payload, "route-policy-benchmark-policy-snapshot")
    return RoutePolicyBenchmarkPolicySnapshot(
        policy_name=str(payload["policyName"]),
        passed=bool(payload.get("passed", False)),
        metrics=_float_mapping(_mapping(payload.get("metrics", {}), "metrics")),
        failed_checks=tuple(str(check) for check in _sequence(payload.get("failedChecks", ()), "failedChecks")),
    )


def route_policy_benchmark_regression_thresholds_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyBenchmarkRegressionThresholds:
    """Rebuild regression thresholds from history JSON."""

    return RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=float(payload.get("maxSuccessRateDrop", 0.0)),
        max_collision_rate_increase=float(payload.get("maxCollisionRateIncrease", 0.0)),
        max_truncation_rate_increase=float(payload.get("maxTruncationRateIncrease", 0.0)),
        max_mean_reward_drop=None if payload.get("maxMeanRewardDrop") is None else float(payload["maxMeanRewardDrop"]),
        require_baseline_policies=bool(payload.get("requireBaselinePolicies", True)),
        fail_on_report_failure=bool(payload.get("failOnReportFailure", True)),
    )


def route_policy_benchmark_regression_check_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyBenchmarkRegressionCheck:
    """Rebuild one regression check from history JSON."""

    _record_type(payload, "route-policy-benchmark-regression-check")
    return RoutePolicyBenchmarkRegressionCheck(
        check_id=str(payload["checkId"]),
        passed=bool(payload.get("passed", False)),
        policy_name=None if payload.get("policyName") is None else str(payload["policyName"]),
        metric_name=None if payload.get("metricName") is None else str(payload["metricName"]),
        baseline_value=_optional_float(payload.get("baselineValue")),
        current_value=_optional_float(payload.get("currentValue")),
        delta=_optional_float(payload.get("delta")),
        allowed_delta=_optional_float(payload.get("allowedDelta")),
        source_path=None if payload.get("sourcePath") is None else str(payload["sourcePath"]),
        message=str(payload.get("message", "")),
    )


def render_route_policy_benchmark_history_markdown(history: RoutePolicyBenchmarkHistoryReport) -> str:
    """Render a compact Markdown summary for benchmark history artifacts."""

    lines = [
        f"# Route Policy Benchmark History: {history.history_id}",
        f"- Status: {'PASS' if history.passed else 'FAIL'}",
        f"- Reports: {len(history.reports)}",
        f"- Baseline: {history.baseline_report.benchmark_id if history.baseline_report else 'n/a'}",
        "",
        "| Source | Benchmark | Pass | Best policy | Policies |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for report in history.reports:
        lines.append(
            "| "
            f"{report.source_path or 'n/a'} | "
            f"{report.benchmark_id} | "
            f"{'yes' if report.passed else 'no'} | "
            f"{report.best_policy_name or 'n/a'} | "
            f"{report.policy_count} |"
        )

    lines.extend(
        [
            "",
            "## Policy Aggregates",
            "",
            "| Policy | Reports | Pass rate | Mean success | Min success | Max collision | Last reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for aggregate in _aggregate_policies(history.reports):
        metrics = _mapping(aggregate.get("metrics", {}), "metrics")
        lines.append(
            "| "
            f"{aggregate['policyName']} | "
            f"{aggregate['reportCount']} | "
            f"{_percent(float(aggregate['passRate']))} | "
            f"{_metric_cell(metrics, 'success-rate', 'mean')} | "
            f"{_metric_cell(metrics, 'success-rate', 'min')} | "
            f"{_metric_cell(metrics, 'collision-rate', 'max')} | "
            f"{_metric_cell(metrics, 'mean-reward', 'last')} |"
        )

    if history.regression_checks:
        lines.extend(
            [
                "",
                "## Regression Gate",
                "",
                "| Check | Policy | Metric | Base | Current | Delta | Pass |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for check in history.regression_checks:
            lines.append(
                "| "
                f"{check.check_id} | "
                f"{check.policy_name or 'n/a'} | "
                f"{check.metric_name or 'n/a'} | "
                f"{_format_optional(check.baseline_value)} | "
                f"{_format_optional(check.current_value)} | "
                f"{_format_optional(check.delta)} | "
                f"{'yes' if check.passed else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy benchmark history CLI."""

    thresholds = RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=float(getattr(args, "max_success_rate_drop")),
        max_collision_rate_increase=float(getattr(args, "max_collision_rate_increase")),
        max_truncation_rate_increase=float(getattr(args, "max_truncation_rate_increase")),
        max_mean_reward_drop=getattr(args, "max_mean_reward_drop", None),
        require_baseline_policies=not bool(getattr(args, "allow_missing_policies", False)),
        fail_on_report_failure=not bool(getattr(args, "allow_report_failures", False)),
    )
    history = build_route_policy_benchmark_history(
        tuple(getattr(args, "report")),
        baseline_report=getattr(args, "baseline_report", None),
        history_id=str(getattr(args, "history_id")),
        thresholds=thresholds,
        metadata={"baselineReport": getattr(args, "baseline_report", None)},
    )
    write_route_policy_benchmark_history_json(getattr(args, "output"), history)
    markdown = render_route_policy_benchmark_history_markdown(history)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"History saved to: {getattr(args, 'output')}")
    if bool(getattr(args, "fail_on_regression", False)) and not history.passed:
        raise SystemExit(2)


def _resolve_baseline_snapshot(
    baseline_report: str | Path | RoutePolicyBenchmarkSnapshot | None,
) -> RoutePolicyBenchmarkSnapshot | None:
    if baseline_report is None:
        return None
    if isinstance(baseline_report, RoutePolicyBenchmarkSnapshot):
        return baseline_report
    return load_route_policy_benchmark_snapshot_json(baseline_report)


def _metric_regression_checks(
    current: RoutePolicyBenchmarkPolicySnapshot,
    baseline: RoutePolicyBenchmarkPolicySnapshot,
    *,
    current_source_path: str | None,
    current_key: str,
    thresholds: RoutePolicyBenchmarkRegressionThresholds,
) -> tuple[RoutePolicyBenchmarkRegressionCheck, ...]:
    checks: list[RoutePolicyBenchmarkRegressionCheck] = []
    gate_specs: tuple[tuple[str, str, float | None], ...] = (
        ("success-rate", "min", thresholds.max_success_rate_drop),
        ("collision-rate", "max", thresholds.max_collision_rate_increase),
        ("truncation-rate", "max", thresholds.max_truncation_rate_increase),
        ("mean-reward", "min", thresholds.max_mean_reward_drop),
    )
    for metric_name, direction, allowed_delta in gate_specs:
        if allowed_delta is None:
            continue
        checks.append(
            _metric_regression_check(
                current,
                baseline,
                metric_name=metric_name,
                direction=direction,
                allowed_delta=float(allowed_delta),
                current_source_path=current_source_path,
                current_key=current_key,
            )
        )
    return tuple(checks)


def _metric_regression_check(
    current: RoutePolicyBenchmarkPolicySnapshot,
    baseline: RoutePolicyBenchmarkPolicySnapshot,
    *,
    metric_name: str,
    direction: str,
    allowed_delta: float,
    current_source_path: str | None,
    current_key: str,
) -> RoutePolicyBenchmarkRegressionCheck:
    check_id = f"{metric_name}-regression:{current_key}:{_slug(current.policy_name)}"
    baseline_value = baseline.metrics.get(metric_name)
    current_value = current.metrics.get(metric_name)
    if baseline_value is None or current_value is None:
        return RoutePolicyBenchmarkRegressionCheck(
            check_id=check_id,
            passed=False,
            policy_name=current.policy_name,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            source_path=current_source_path,
            message=f"metric {metric_name} is missing from baseline or current report",
        )

    delta = float(current_value) - float(baseline_value)
    if direction == "min":
        passed = current_value >= baseline_value - allowed_delta
        message = (
            f"{metric_name} may drop by at most {allowed_delta:.6g}; "
            f"baseline={baseline_value:.6g}, current={current_value:.6g}"
        )
    elif direction == "max":
        passed = current_value <= baseline_value + allowed_delta
        message = (
            f"{metric_name} may increase by at most {allowed_delta:.6g}; "
            f"baseline={baseline_value:.6g}, current={current_value:.6g}"
        )
    else:
        raise ValueError(f"unsupported metric direction: {direction}")
    return RoutePolicyBenchmarkRegressionCheck(
        check_id=check_id,
        passed=passed,
        policy_name=current.policy_name,
        metric_name=metric_name,
        baseline_value=float(baseline_value),
        current_value=float(current_value),
        delta=delta,
        allowed_delta=float(allowed_delta),
        source_path=current_source_path,
        message=message,
    )


def _aggregate_policies(reports: Sequence[RoutePolicyBenchmarkSnapshot]) -> list[dict[str, Any]]:
    series: dict[str, list[RoutePolicyBenchmarkPolicySnapshot]] = {}
    for report in reports:
        for policy in report.policies:
            series.setdefault(policy.policy_name, []).append(policy)

    aggregates: list[dict[str, Any]] = []
    for policy_name, snapshots in series.items():
        metric_names = sorted({metric_name for snapshot in snapshots for metric_name in snapshot.metrics})
        metrics = {
            metric_name: _metric_series(
                [snapshot.metrics[metric_name] for snapshot in snapshots if metric_name in snapshot.metrics]
            )
            for metric_name in metric_names
        }
        aggregates.append(
            {
                "policyName": policy_name,
                "reportCount": len(snapshots),
                "passedCount": sum(1 for snapshot in snapshots if snapshot.passed),
                "passRate": _safe_rate(sum(1 for snapshot in snapshots if snapshot.passed), len(snapshots)),
                "metrics": metrics,
            }
        )
    return aggregates


def _metric_series(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "first": 0.0, "last": 0.0, "delta": 0.0}
    first = float(values[0])
    last = float(values[-1])
    return {
        "count": float(len(values)),
        "min": min(float(value) for value in values),
        "max": max(float(value) for value in values),
        "mean": sum(float(value) for value in values) / len(values),
        "first": first,
        "last": last,
        "delta": last - first,
    }


def _snapshot_key(snapshot: RoutePolicyBenchmarkSnapshot) -> str:
    if snapshot.source_path:
        return _slug(Path(snapshot.source_path).stem)
    return _slug(snapshot.benchmark_id)


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


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


def _non_negative_float(value: float, field_name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"


def _metric_cell(metrics: Mapping[str, Any], metric_name: str, key: str) -> str:
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        return "n/a"
    value = metric.get(key)
    return "n/a" if value is None else _format_float(float(value))


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"
