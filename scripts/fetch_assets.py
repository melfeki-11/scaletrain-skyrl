"""Fetch/build the pinned, untracked assets: model snapshot, GSM8K parquet splits, vllm-router wheel.

Run from the repo root with a Python that has: huggingface_hub, datasets, pyarrow, pyyaml (see requirements-local.txt).
"""
import hashlib
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from q35.config import load  # noqa: E402

ROUTER_WHEEL = 'vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_35_x86_64.whl'
ROUTER_SHA256 = '4a1fa0bc7708e6498a12ce67f13a1c14e30e6307aeeeddd345448e8264a4718b'


def main():
    cfg = load()
    from huggingface_hub import snapshot_download
    snapshot_download(cfg['model']['id'], revision=cfg['model']['revision'], local_dir=cfg['model']['local_path'])
    from q35.data import prepare_dataset
    prepare_dataset(cfg)  # pinned dataset revision + seeded train/validation split; official test stays untouched
    runtime = Path('assets/runtime'); runtime.mkdir(parents=True, exist_ok=True)
    wheel = runtime / ROUTER_WHEEL
    if not wheel.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'download', 'vllm-router==0.1.14.post1', '--no-deps', '--only-binary=:all:',
                        '--platform', 'manylinux_2_35_x86_64', '--python-version', '3.12', '--abi', 'abi3', '-d', str(runtime)], check=True)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != ROUTER_SHA256:
        raise SystemExit('vllm-router wheel hash mismatch')
    print('assets ready')


if __name__ == '__main__':
    main()
