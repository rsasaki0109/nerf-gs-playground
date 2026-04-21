#!/usr/bin/env python3
"""Collect artifact and metric summaries for planned MCD quality runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.experiments.mcd_quality_plan import (  # noqa: E402
    MCDQualityPlanContext,
    build_mcd_quality_plan,
    collect_mcd_quality_results,
    default_mcd_quality_profiles,
    render_quality_benchmark_markdown,
    render_quality_report_json,
    render_quality_report_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default="data/mcd/ntu_day_02", help="MCD session directory")
    parser.add_argument("--output-root", default="outputs/mcd_quality", help="Root directory for planned outputs")
    parser.add_argument(
        "--calibration",
        default="data/mcd/calibration_atv.yaml",
        help="Official MCD ATV calibration YAML path",
    )
    parser.add_argument(
        "--asset-dir", default="outputs/mcd_quality/assets", help="Directory for planned .splat exports"
    )
    parser.add_argument("--python", default="python3", help="Python executable used in planned commands")
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH used in planned commands")
    parser.add_argument("--start-offset-sec", type=float, default=35.0, help="GNSS/image/LiDAR warm-up trim")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_mcd_quality_profiles()],
        help="Only collect the named profile. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=["markdown", "json", "benchmark"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional path to write the rendered report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    context = MCDQualityPlanContext(
        session_dir=args.session_dir,
        output_root=args.output_root,
        calibration_path=args.calibration,
        asset_dir=args.asset_dir,
        python_executable=args.python,
        pythonpath=args.pythonpath,
        start_offset_sec=args.start_offset_sec,
    )
    profiles = default_mcd_quality_profiles()
    if args.profile:
        wanted = set(args.profile)
        profiles = tuple(profile for profile in profiles if profile.name in wanted)
    report = collect_mcd_quality_results(build_mcd_quality_plan(context, profiles=profiles))
    if args.format == "json":
        rendered = render_quality_report_json(report)
    elif args.format == "benchmark":
        rendered = render_quality_benchmark_markdown(report)
    else:
        rendered = render_quality_report_markdown(report)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
