#!/usr/bin/env python3
"""Read only exact-job Kubernetes lifecycle metadata without printing pod specs/logs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def pod_summary(pod, job_id):
    meta = pod.get("metadata", {})
    if meta.get("labels", {}).get("scaletrain/job_id") != job_id:
        raise ValueError("Pod label does not match requested job ID.")
    spec = pod.get("spec", {}); status = pod.get("status", {})
    requests = [{"container": c.get("name"), "requests": {k: v for k, v in c.get("resources", {}).get("requests", {}).items()
                    if k in {"cpu", "memory", "ephemeral-storage", "nvidia.com/gpu", "vpc.amazonaws.com/efa"}}}
                for c in spec.get("containers", [])]
    init_requests = [{"container": c.get("name"), "requests": {k: v for k, v in c.get("resources", {}).get("requests", {}).items()
                    if k in {"cpu", "memory", "ephemeral-storage", "nvidia.com/gpu", "vpc.amazonaws.com/efa"}}}
                for c in spec.get("initContainers", [])]
    conditions = [{k: c.get(k) for k in ["type", "status", "lastTransitionTime"]}
                  for c in status.get("conditions", []) if c.get("type") in {"PodScheduled", "Initialized", "Ready", "ContainersReady"}]
    containers = []
    for group in ["initContainerStatuses", "containerStatuses"]:
        for container in status.get(group, []):
            states = {}
            for source in ["state", "lastState"]:
                for kind in ["running", "terminated", "waiting"]:
                    value = container.get(source, {}).get(kind)
                    if value is not None:
                        states[source] = {"kind": kind, **{k: value.get(k) for k in ["startedAt", "finishedAt", "exitCode"] if k in value}}
            containers.append({"name": container.get("name"), "restart_count": container.get("restartCount"), **states})
    return {"name": meta.get("name"), "uid": meta.get("uid"), "created_at": meta.get("creationTimestamp"),
            "deleted_at": meta.get("deletionTimestamp"), "node": spec.get("nodeName"), "phase": status.get("phase"),
            "pod_start_time": status.get("startTime"), "conditions": conditions,
            "container_states": containers, "requests": requests, "init_requests": init_requests}


def job_summary(job):
    meta = job.get("metadata", {}); status = job.get("status", {})
    return {"name": meta.get("name"), "uid": meta.get("uid"), "created_at": meta.get("creationTimestamp"),
            "start_time": status.get("startTime"), "completion_time": status.get("completionTime"),
            "active": status.get("active"), "succeeded": status.get("succeeded"), "failed": status.get("failed"),
            "conditions": [{k: c.get(k) for k in ["type", "status", "lastTransitionTime"]}
                           for c in status.get("conditions", []) if c.get("type") in {"Complete", "Failed", "Suspended"}]}


def collect(context, namespace, job_id):
    if not re.fullmatch(r"job_[a-z0-9]+", job_id):
        raise ValueError("Expected exact ScaleTrain job ID.")
    base = ["kubectl", f"--context={context}", f"--namespace={namespace}", "get"]
    def request(args):
        response = subprocess.run(base + args + ["-o", "json"], check=True, capture_output=True, text=True, timeout=40)
        return json.loads(response.stdout)
    pods = request(["pods", f"--selector=scaletrain/job_id={job_id}"]).get("items", [])
    summaries = [pod_summary(pod, job_id) for pod in pods]
    owners = {(ref["name"], ref["uid"]) for pod in pods for ref in pod.get("metadata", {}).get("ownerReferences", []) if ref.get("kind") == "Job"}
    jobs = []
    for name, uid in sorted(owners):
        job = request(["job", name])
        if job.get("metadata", {}).get("uid") != uid:
            raise ValueError("Job owner UID changed; refresh lifecycle receipt.")
        jobs.append(job_summary(job))
    return {"job_id": job_id, "observed_at": datetime.now(timezone.utc).isoformat(), "context": context,
            "namespace": namespace, "pods": summaries, "jobs": jobs,
            "allocation_start_verified": False,
            "note": "Pod/job times are lifecycle evidence, not Kueue admission. Preserve historical receipts; current objects may omit preempted or deleted intervals."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True); parser.add_argument("--namespace", required=True)
    parser.add_argument("--job-id", required=True); parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    target = Path(args.receipt)
    if target.exists():
        raise ValueError("Use a new receipt filename to preserve historical lifecycle evidence.")
    result = collect(args.context, args.namespace, args.job_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"receipt": str(target), "pods": len(result["pods"]), "jobs": len(result["jobs"]),
                      "distinct_nodes": sorted({pod["node"] for pod in result["pods"] if pod["node"]}),
                      "allocation_start_verified": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": "Lifecycle read failed; verify exact job ID, context, namespace, RBAC and receipt path."}))
        raise SystemExit(1)
