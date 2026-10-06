#!/usr/bin/env python3
"""Build assets/overlay: the uv.lock-exact Modal + Harbor packages the SkyRL training image lacks, for linux x86_64 / CPython 3.12.

    python tools/build_overlay.py            # -> assets/overlay (~125 MB); staged with Modal/Harbor experiments, first on PYTHONPATH

Why: the image's /opt/skyrl-venv is root-owned and the worker runs as uid 1000, so nothing can be pip-installed at job time; the
worker needs no network or uv this way. The bundled `claude` CLI (276 MB) and bin/ scripts are stripped (unused by trials).
"""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRIP = ('claude_agent_sdk/_bundled/claude', 'bin')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', default=str(ROOT / 'assets/overlay'))
    p.add_argument('--force', action='store_true', help='replace an existing overlay')
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        if not a.force:
            raise SystemExit(f'{target} exists (pass --force to rebuild)')
        shutil.rmtree(target)
    subprocess.run(['uv', 'pip', 'install', '--target', str(target), '--no-deps', '--python-platform', 'x86_64-manylinux_2_28',
                    '--python-version', '3.12', '-r', str(ROOT / 'tools/overlay-requirements.txt')], check=True)
    for rel in STRIP:
        path = target / rel
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) / 1e6
    print(f'{{"overlay": "{target}", "megabytes": {size:.1f}}}')


if __name__ == '__main__':
    main()
