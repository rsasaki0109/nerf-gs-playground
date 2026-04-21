"""Capture README table thumbnails from local ``docs/splat.html`` (Playwright).

Hides the info blurb and scene picker so the canvas fills the shot — reads
clearer at small GitHub-rendered sizes than a full-page crop. The production
scene list and output filenames come from ``docs/scenes-list.json``.

  # WebGL splats need a real GPU context — headed mode (default) is reliable on Linux:
  DISPLAY=:0 python3 scripts/capture_readme_splat_previews.py

  Force headless (often black canvas): PLAYWRIGHT_HEADLESS=1 python3 ...

Requires: ``pip install playwright`` and ``playwright install chromium``.
"""

from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT_DIR = DOCS / "images" / "demo-sweep"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pages_scene_manifest import capture_scene_specs  # noqa: E402


SCENES = capture_scene_specs(DOCS)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=[s[0] for s in SCENES],
        default=None,
        help="Capture a single scene by output stem (default: all)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless Chromium (frequently yields empty WebGL for splat.html; prefer headed).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for PNG captures (default: docs/images/demo-sweep).",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=8000,
        help="Milliseconds to wait after each scene load before capturing (default: 8000).",
    )
    args = parser.parse_args()
    use_headless = args.headless or os.environ.get("PLAYWRIGHT_HEADLESS", "").lower() in ("1", "true", "yes")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("install playwright: pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    scenes = SCENES
    if args.only:
        scenes = [x for x in SCENES if x[0] == args.only]
        if not scenes:
            return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(DOCS),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.6)
    hide_ui = """
    #info, #scenePickerRow, #instructions, #progress, #message, #spinner,
    #quality, #caminfo { display: none !important; }
    body { margin: 0; overflow: hidden; background: #000; }
    """

    launch_args = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-gpu-blocklist"]
    if use_headless:
        launch_args.extend(["--use-gl=angle", "--use-angle=swiftshader"])

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=use_headless, args=launch_args)
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
            )
            page = ctx.new_page()
            for stem, rel in scenes:
                if not (DOCS / rel).is_file():
                    print(f"skip {stem}: missing docs/{rel}", file=sys.stderr)
                    continue
                url = f"http://127.0.0.1:{port}/splat.html?url={rel}"
                page.goto(url, wait_until="commit", timeout=120_000)
                page.add_style_tag(content=hide_ui)
                page.wait_for_timeout(args.wait_ms)
                out = out_dir / f"{stem}.png"
                box = page.locator("#canvas").bounding_box()
                if not box:
                    print(f"skip {stem}: no #canvas bbox", file=sys.stderr)
                    continue
                # Element screenshots on WebGL canvases often time out in headless Chromium;
                # a one-shot viewport clip is reliable.
                # Playwright's Page.screenshot can hang on continuously redrawn WebGL canvases;
                # CDP capture returns the current framebuffer without waiting for "idle".
                cdp = page.context.new_cdp_session(page)
                cdp.send("Page.enable")
                shot = cdp.send(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "clip": {
                            "x": box["x"],
                            "y": box["y"],
                            "width": box["width"],
                            "height": box["height"],
                            "scale": 1,
                        },
                    },
                )
                raw = base64.b64decode(shot["data"])
                out.write_bytes(raw)
                sz = len(raw)
                print(f"wrote {out} ({sz} bytes)")
                if sz < 50_000:
                    print(
                        f"warning: {stem} looks like an empty/black WebGL grab "
                        "(try without --headless and a real DISPLAY).",
                        file=sys.stderr,
                    )
            ctx.close()
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=3)
        except subprocess.TimeoutExpired:
            srv.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
