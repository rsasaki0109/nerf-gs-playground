#!/usr/bin/env python3
"""Generate the Physical AI simulation scene catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.sim import (  # noqa: E402
    DEFAULT_SITE_URL,
    load_simulation_catalog_from_scene_picker,
    render_simulation_catalog_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-picker",
        default="docs/scenes-list.json",
        help="Input viewer scene picker catalog",
    )
    parser.add_argument(
        "--output",
        default="docs/sim-scenes.json",
        help="Output JSON path. Use '-' for stdout.",
    )
    parser.add_argument(
        "--site-url",
        default=DEFAULT_SITE_URL,
        help="Base GitHub Pages URL used to build viewer deep links",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    catalog = load_simulation_catalog_from_scene_picker(REPO_ROOT / args.scene_picker, site_url=args.site_url)
    rendered = render_simulation_catalog_json(catalog)

    if args.output == "-":
        print(rendered, end="")
        return

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
