"""Resolve external visual SLAM output files without importing those projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gs_sim2real.preprocess.external_slam_artifacts.profiles import PROFILES, normalize_system


_SKIP_TEXT_NAMES = ("readme", "license", "config", "metrics", "results", "log")


@dataclass(frozen=True, slots=True)
class ExternalSLAMArtifacts:
    """Resolved files ready for conversion through the existing trajectory importer."""

    system: str
    trajectory_path: Path
    trajectory_format: str
    pointcloud_path: Path | None = None
    pinhole_calib_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ExternalSLAMCandidateTrace:
    """One candidate-pattern probe performed while resolving an external artifact."""

    pattern: str
    match_count: int
    selected_path: Path | None
    skipped_paths: tuple[Path, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ExternalSLAMFileResolutionTrace:
    """Trace of how one artifact role was resolved."""

    role: str
    explicit_path: Path | None
    base_dir: Path | None
    candidate_patterns: tuple[str, ...]
    selected_path: Path | None
    candidate_traces: tuple[ExternalSLAMCandidateTrace, ...]
    reason: str


def resolve_external_slam_artifacts(
    *,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> ExternalSLAMArtifacts:
    """Resolve external SLAM output paths without importing the external project."""

    system_key = normalize_system(system)
    profile = PROFILES[system_key]
    base_dir = _optional_existing_dir(artifact_dir, role="external SLAM output")

    resolved_trajectory = _resolve_required_file(
        explicit=trajectory_path,
        base_dir=base_dir,
        candidates=profile.trajectory_candidates,
        role=f"{profile.display_name} trajectory",
    )
    resolved_pointcloud = _resolve_optional_file(
        explicit=pointcloud_path,
        base_dir=base_dir,
        candidates=profile.pointcloud_candidates,
        role=f"{profile.display_name} point cloud",
    )
    resolved_calib = _resolve_optional_file(
        explicit=pinhole_calib_path,
        base_dir=base_dir,
        candidates=(),
        role="PINHOLE calibration",
    )

    return ExternalSLAMArtifacts(
        system=system_key,
        trajectory_path=resolved_trajectory,
        trajectory_format=trajectory_format or profile.default_trajectory_format,
        pointcloud_path=resolved_pointcloud,
        pinhole_calib_path=resolved_calib,
    )


def trace_external_slam_file_resolution(
    *,
    explicit: str | Path | None,
    base_dir: str | Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> ExternalSLAMFileResolutionTrace:
    """Trace file resolution without raising for missing paths."""

    explicit_path = Path(explicit) if explicit not in (None, "") else None
    base_path = Path(base_dir) if base_dir not in (None, "") else None
    if explicit_path is not None:
        return _trace_explicit_file(explicit_path, base_dir=base_path, candidates=candidates, role=role)
    if base_path is None:
        return ExternalSLAMFileResolutionTrace(
            role=role,
            explicit_path=None,
            base_dir=None,
            candidate_patterns=candidates,
            selected_path=None,
            candidate_traces=(),
            reason="no_base_dir",
        )
    if not base_path.exists():
        return ExternalSLAMFileResolutionTrace(
            role=role,
            explicit_path=None,
            base_dir=base_path,
            candidate_patterns=candidates,
            selected_path=None,
            candidate_traces=(),
            reason="base_dir_missing",
        )
    if not base_path.is_dir():
        return ExternalSLAMFileResolutionTrace(
            role=role,
            explicit_path=None,
            base_dir=base_path,
            candidate_patterns=candidates,
            selected_path=None,
            candidate_traces=(),
            reason="base_dir_not_directory",
        )
    if not candidates:
        return ExternalSLAMFileResolutionTrace(
            role=role,
            explicit_path=None,
            base_dir=base_path,
            candidate_patterns=candidates,
            selected_path=None,
            candidate_traces=(),
            reason="no_candidates",
        )

    selected_path: Path | None = None
    traces: list[ExternalSLAMCandidateTrace] = []
    for pattern in candidates:
        matches, skipped = _candidate_matches_with_skips(base_path, pattern)
        pattern_selected = matches[0] if selected_path is None and matches else None
        if pattern_selected is not None:
            selected_path = pattern_selected
            reason = "selected"
        elif matches:
            reason = "matched_after_selection"
        elif skipped:
            reason = "only_skipped_non_trajectory_text"
        else:
            reason = "no_match"
        traces.append(
            ExternalSLAMCandidateTrace(
                pattern=pattern,
                match_count=len(matches),
                selected_path=pattern_selected,
                skipped_paths=tuple(skipped),
                reason=reason,
            )
        )

    return ExternalSLAMFileResolutionTrace(
        role=role,
        explicit_path=None,
        base_dir=base_path,
        candidate_patterns=candidates,
        selected_path=selected_path,
        candidate_traces=tuple(traces),
        reason="selected_candidate" if selected_path is not None else "no_candidate_match",
    )


def _optional_existing_dir(path: str | Path | None, *, role: str) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{role} directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{role} path is not a directory: {candidate}")
    return candidate


def _resolve_required_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path:
    resolved = _resolve_optional_file(explicit=explicit, base_dir=base_dir, candidates=candidates, role=role)
    if resolved is not None:
        return resolved
    if base_dir is None:
        raise ValueError(f"{role} is required. Pass --trajectory or --external-slam-output.")
    raise FileNotFoundError(f"Could not find {role} under {base_dir}")


def _resolve_optional_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path | None:
    if explicit not in (None, ""):
        return _resolve_explicit_file(Path(explicit), base_dir=base_dir, role=role)
    if base_dir is None:
        return None
    for pattern in candidates:
        matches = _candidate_matches(base_dir, pattern)
        if matches:
            return matches[0]
    return None


def _resolve_explicit_file(path: Path, *, base_dir: Path | None, role: str) -> Path:
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_file():
                raise FileNotFoundError(f"{role} path is not a file: {candidate}")
            return candidate
    raise FileNotFoundError(f"{role} file not found: {path}")


def _candidate_matches(base_dir: Path, pattern: str) -> list[Path]:
    matches, _skipped = _candidate_matches_with_skips(base_dir, pattern)
    return matches


def _candidate_matches_with_skips(base_dir: Path, pattern: str) -> tuple[list[Path], list[Path]]:
    raw_matches = sorted(p for p in base_dir.rglob(pattern) if p.is_file())
    if not (pattern.endswith(".txt") or pattern == "*.txt"):
        return raw_matches, []
    skipped = [p for p in raw_matches if _looks_like_non_trajectory_text(p)]
    skipped_set = set(skipped)
    matches = [p for p in raw_matches if p not in skipped_set]
    return matches, skipped


def _trace_explicit_file(
    path: Path,
    *,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> ExternalSLAMFileResolutionTrace:
    checked = [path]
    if base_dir is not None and not path.is_absolute():
        checked.append(base_dir / path)

    selected_path: Path | None = None
    skipped_paths: list[Path] = []
    reason = "explicit_path_missing"
    for candidate in checked:
        if not candidate.exists():
            continue
        if candidate.is_file():
            selected_path = candidate
            reason = "explicit_path_found"
            break
        skipped_paths.append(candidate)
        reason = "explicit_path_not_file"

    trace = ExternalSLAMCandidateTrace(
        pattern=str(path),
        match_count=1 if selected_path is not None else 0,
        selected_path=selected_path,
        skipped_paths=tuple(skipped_paths),
        reason=reason,
    )
    return ExternalSLAMFileResolutionTrace(
        role=role,
        explicit_path=path,
        base_dir=base_dir,
        candidate_patterns=candidates,
        selected_path=selected_path,
        candidate_traces=(trace,),
        reason=reason,
    )


def _looks_like_non_trajectory_text(path: Path) -> bool:
    name = path.name.lower()
    return any(skip in name for skip in _SKIP_TEXT_NAMES)


__all__ = [
    "ExternalSLAMArtifacts",
    "ExternalSLAMCandidateTrace",
    "ExternalSLAMFileResolutionTrace",
    "resolve_external_slam_artifacts",
    "trace_external_slam_file_resolution",
]
