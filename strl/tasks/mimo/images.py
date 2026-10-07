"""Build every distinct exported-task image once in Modal (build-cache warm), concurrency
limited, with Docker Hub registry rate-limit awareness, plus (unimplemented-by-default, command
construction only) an ECR-mirroring fallback for when anonymous pulls get blocked.

Used by ``tools/prebuild_mimo_images.py``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ECR_ACCOUNT = "692474966980"  # base/infra.yaml's image account, reused for the mirror repo
ECR_REGION = "us-west-2"
ECR_REPO_PREFIX = "mimo"

_RATE_LIMIT_FLOOR = 10  # stop and report if RateLimit-Remaining drops below this


# ------------------------------------------------------------------------- Docker Hub HEAD probe


def registry_head(image: str, *, timeout_sec: float = 15.0) -> dict[str, Any]:
    """HEAD the manifest of a ``docker.io/<repo>:<tag>`` image (anonymous token flow) and
    report the rate-limit headers Docker Hub returns. Returns
    ``{"ok", "remaining", "limit", "status"}``; ``remaining``/``limit`` are ``None`` when the
    registry didn't send the headers (e.g. a non-Docker-Hub registry)."""
    ref = image.removeprefix("docker.io/")
    if "/" not in ref:
        ref = f"library/{ref}"
    repo, _, tag = ref.rpartition(":")
    tag = tag or "latest"

    token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    with urllib.request.urlopen(token_url, timeout=timeout_sec) as resp:
        token = json.loads(resp.read())["token"]

    manifest_url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(manifest_url, method="HEAD")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.docker.distribution.manifest.v2+json")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            headers = resp.headers
            status = resp.status
    except urllib.error.HTTPError as e:
        headers = e.headers
        status = e.code

    remaining = headers.get("RateLimit-Remaining")  # e.g. "100;w=21600"
    limit = headers.get("RateLimit-Limit")
    remaining_n = int(remaining.split(";")[0]) if remaining else None
    return {"ok": status < 400, "status": status, "remaining": remaining_n, "limit": limit}


# --------------------------------------------------------------------------------- Modal build


def build_task_image(task_dir: Path, *, app_name: str = "mimo-prebuild", timeout_sec: int = 1800) -> dict[str, Any]:
    """Build the task's ``environment/Dockerfile`` in Modal (warms Harbor's build cache for
    this exact image), start a tiny sandbox on it to prove it boots and ``tmux -V`` works, and
    terminate. Returns ``{"image", "seconds", "ok", "error"}``."""
    import modal

    strl_json = json.loads((task_dir / "strl.json").read_text())
    image_ref = strl_json["image"]
    started = time.monotonic()
    try:
        image = modal.Image.from_dockerfile(task_dir / "environment" / "Dockerfile", context_dir=task_dir / "environment")
        app = modal.App.lookup(app_name, create_if_missing=True)
        sb = modal.Sandbox.create(app=app, image=image, cpu=1, memory=1024, timeout=120, block_network=True)
        try:
            proc = sb.exec("tmux", "-V")
            tmux_version = proc.stdout.read().strip()
            proc.wait()
            tmux_ok = proc.returncode == 0
        finally:
            sb.terminate()
        seconds = round(time.monotonic() - started, 1)
        return {"image": image_ref, "seconds": seconds, "ok": tmux_ok, "tmux_version": tmux_version, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"image": image_ref, "seconds": round(time.monotonic() - started, 1), "ok": False, "error": f"{type(e).__name__}: {e}"[:500]}


def prebuild_images(
    task_dirs: list[Path], *, concurrency: int = 8, app_name: str = "mimo-prebuild", rate_limit_floor: int = _RATE_LIMIT_FLOOR
) -> dict[str, Any]:
    """Build each DISTINCT image among ``task_dirs`` once, concurrency-limited, stopping (and
    reporting) if a Docker Hub HEAD shows ``RateLimit-Remaining < rate_limit_floor`` before a
    build starts. Returns ``{"results": [...], "stopped_early": bool, "reason": str | None}``."""
    import asyncio

    seen: dict[str, Path] = {}
    for td in task_dirs:
        strl_json = json.loads((Path(td) / "strl.json").read_text())
        seen.setdefault(strl_json["image"], Path(td))
    distinct = list(seen.items())

    results: list[dict] = []
    stopped_early = False
    reason = None

    async def _run_one(image_ref: str, td: Path, sem: "asyncio.Semaphore") -> None:
        async with sem:
            if image_ref.startswith("docker.io/"):
                try:
                    head = registry_head(image_ref)
                    if head.get("remaining") is not None and head["remaining"] < rate_limit_floor:
                        nonlocal_stop(image_ref, head)
                        return
                except Exception:  # noqa: BLE001 - a HEAD probe failure must not block the build
                    pass
            result = await asyncio.to_thread(build_task_image, td, app_name=app_name)
            results.append(result)

    def nonlocal_stop(image_ref: str, head: dict) -> None:
        nonlocal stopped_early, reason
        stopped_early = True
        reason = f"RateLimit-Remaining={head['remaining']} < {rate_limit_floor} before building {image_ref}"

    async def _run_all() -> None:
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*(_run_one(image_ref, td, sem) for image_ref, td in distinct))

    asyncio.run(_run_all())
    return {"results": results, "stopped_early": stopped_early, "reason": reason, "distinct_images": len(distinct)}


# --------------------------------------------------------------------- ECR mirror (fallback)


def ecr_repo_uri(image_ref: str, *, account: str = ECR_ACCOUNT, region: str = ECR_REGION, prefix: str = ECR_REPO_PREFIX) -> str:
    """Deterministic private-ECR repo name for a Docker Hub image ref, so the same source image
    always mirrors to the same target: ``<account>.dkr.ecr.<region>.amazonaws.com/<prefix>/<repo-tag-with-/_-only>``."""
    tag_part = image_ref.removeprefix("docker.io/").replace("/", "__").replace(":", "-")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{prefix}/{tag_part}"


def ecr_mirror_commands(
    image_ref: str, digest: str, *, account: str = ECR_ACCOUNT, region: str = ECR_REGION, profile: str = "production-developer"
) -> list[list[str]]:
    """Construct (but never execute) the shell command sequence to copy ``image_ref@digest``
    into a private ECR repo by digest, using the ``production-developer`` AWS profile. Returns
    a list of argv lists (``subprocess.run``-ready); the caller decides whether/when to run
    them -- per the task's budget rules, this requires explicit sign-off before ever running
    for real.
    """
    repo_uri = ecr_repo_uri(image_ref, account=account, region=region)
    repo_name = repo_uri.split("/", 1)[1]
    source = f"{image_ref.removeprefix('docker.io/')}@{digest}"
    return [
        ["aws", "ecr", "describe-repositories", "--repository-names", repo_name, "--region", region, "--profile", profile],
        ["aws", "ecr", "create-repository", "--repository-name", repo_name, "--region", region, "--profile", profile],
        ["docker", "pull", source],
        ["docker", "tag", source, f"{repo_uri}:{digest.removeprefix('sha256:')[:16]}"],
        [
            "bash", "-c",
            f"aws ecr get-login-password --region {region} --profile {profile} | "
            f"docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com",
        ],
        ["docker", "push", f"{repo_uri}:{digest.removeprefix('sha256:')[:16]}"],
    ]
