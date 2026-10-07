#!/usr/bin/env python3
"""Build every distinct image among a selection of exported MiMo task dirs, once, in Modal
(warms Harbor's build cache), concurrency-limited, with Docker Hub rate-limit awareness.

    python tools/prebuild_mimo_images.py <data_dir>/tasks/canary --concurrency 8

Writes evidence/gates/mimo-images.json = ``{"image", "seconds", "ok"}`` per distinct image,
plus the rate-limit-stop receipt when applicable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assets" / "overlay"))  # harbor + modal SDKs

from strl.tasks.mimo import images  # noqa: E402


def discover_task_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("strl.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_root", type=Path, help="directory containing exported task dirs (searched recursively for strl.json)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--app-name", default="mimo-prebuild")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "gates" / "mimo-images.json")
    args = ap.parse_args()

    task_dirs = discover_task_dirs(args.tasks_root)
    if not task_dirs:
        print(f"no strl.json found under {args.tasks_root}", file=sys.stderr)
        return 1
    print(f"found {len(task_dirs)} task dir(s); building distinct images with concurrency={args.concurrency}")

    receipt = images.prebuild_images(task_dirs, concurrency=args.concurrency, app_name=args.app_name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"wrote {args.out}")

    for r in receipt["results"]:
        status = "OK" if r["ok"] else f"FAIL ({r.get('error')})"
        print(f"  {r['image']}: {status} ({r['seconds']}s)")
    if receipt["stopped_early"]:
        print(f"STOPPED EARLY: {receipt['reason']}", file=sys.stderr)
        return 2
    return 0 if all(r["ok"] for r in receipt["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
