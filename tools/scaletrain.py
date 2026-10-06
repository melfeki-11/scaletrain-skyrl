#!/usr/bin/env python3
"""Headless ScaleTrain API, using the installed scale-train CLI interpreter and schema (vendored from the scaletrain-services skill).

submit records intent BEFORE the POST and refuses a name that already exists, so an ambiguous response never causes a duplicate
launch. If submit ever errors, run `list --user-id <you>` and reconcile by exact job name; never resubmit blindly.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys


TERMINAL = {"COMPLETED", "FAILED", "CANCELED", "CANCELLED"}


def cli_python():
    executable = shutil.which("scale-train")
    if not executable:
        raise RuntimeError("scale-train is missing; install with the team's authenticated CodeArtifact setup.")
    line = Path(executable).read_text().splitlines()[0]
    interpreter = line.removeprefix("#!").strip()
    if not line.startswith("#!/") or " " in interpreter or not Path(interpreter).is_file():
        raise RuntimeError("Cannot resolve CLI Python; use the interpreter containing scaletrain_cli.")
    return interpreter


def ensure_cli():
    import importlib.util
    if importlib.util.find_spec("scaletrain_cli") is None:
        os.execv(cli_python(), [cli_python(), str(Path(__file__).resolve()), *sys.argv[1:]])


def validate(data):
    if data.get("job_type") != "k8s-preset":
        raise ValueError("This submit helper supports EKS k8s-preset; use a verified project launcher for GKE.")
    required = ["job_name", "image", "team", "product", "command", "preset_instance_type"]
    if any(not data.get(k) for k in required):
        raise ValueError("Missing required job fields.")
    if not isinstance(data["command"], list) or not all(isinstance(v, str) for v in data["command"]):
        raise ValueError("command must be a list of strings.")
    if type(data.get("num_instances")) is not int or data["num_instances"] < 1:
        raise ValueError("num_instances must be a positive integer.")
    if type(data.get("active_deadline_seconds")) is not int or not 1 <= data["active_deadline_seconds"] <= 604800:
        raise ValueError("Set an explicit positive deadline no longer than 7 days.")
    if data.get("priority") not in {"high", "normal", "low"}:
        raise ValueError("priority must be high, normal, or low.")
    if type(data.get("allow_borrowing")) is not bool:
        raise ValueError("Set allow_borrowing explicitly.")
    if re.search(r"\$\{[^}]+\}", json.dumps(data)):
        raise ValueError("Unresolved job placeholders.")
    allowed = set(required) | {"job_type", "num_instances", "active_deadline_seconds", "priority", "allow_borrowing"}
    if set(data) - allowed:
        raise ValueError("Unknown job fields; check the installed schema.")
    for value in data["command"]:
        if re.search(r"(?:WANDB_API_KEY|SERP_API_KEY|SERPER_API_KEY|LITELLM_API_KEY|HIL_BENCH\w*)\s*=", value):
            raise ValueError("Pass a secret locator, never literal credentials in commands.")
    from scaletrain_cli.domain.entities.user_configs.training.config_models import TrainingConfigModel
    return TrainingConfigModel.from_dict(data)


def summary(job):
    d = job.to_dict()
    c = d.get("job_config", {})
    return {k: d.get(k) for k in ["id", "name", "status", "created_by", "team", "product", "created_at", "queue_position"]} | {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status_detail": classify_status_detail(d.get("status_detail")),
        "allocation_timestamps_available": False,
        "runtime": {k: c.get(k) for k in ["image", "job_config_type", "num_instances", "preset_instance_type", "active_deadline_seconds", "priority", "allow_borrowing"]}}


def classify_status_detail(detail):
    """Return fixed categories; raw service messages may contain user log content."""
    text = str(detail or "").lower()
    if not text:
        return "absent"
    if "single network domain" in text:
        return "waiting_for_network_domain_capacity"
    if "preempt" in text:
        return "preemption_reported"
    if "image" in text and any(word in text for word in ["pull", "failed", "error"]):
        return "image_issue_reported"
    if "deadline" in text or "timeout" in text:
        return "deadline_or_timeout_reported"
    return "present_private_detail_requires_inspection"


def progress_summary(entries):
    """Timeline metadata only; progress is not scheduler allocation evidence."""
    events = []
    for entry in entries:
        data = entry.to_dict()
        events.append({"creation_time": data.get("creation_time"), "payload_omitted": True})
    return {"entries": len(events), "events": events, "allocation_timestamps_available": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("list"); q.add_argument("--user-id", required=True); q.add_argument("--limit", type=int, default=50)
    q = sub.add_parser("get"); q.add_argument("job_id")
    q = sub.add_parser("progress"); q.add_argument("job_id")
    q = sub.add_parser("queue"); q.add_argument("instance_type"); q.add_argument("--limit", type=int, default=20)
    q = sub.add_parser("validate"); q.add_argument("job_file")
    q = sub.add_parser("submit"); q.add_argument("job_file"); q.add_argument("--user-id", required=True)
    q.add_argument("--receipt", required=True)
    q = sub.add_parser("cancel"); q.add_argument("job_id"); q.add_argument("--user-id", required=True)
    args = p.parse_args()
    ensure_cli()
    # The CLI gateway prints API exception bodies. Replace its context manager locally
    # so helpers keep machine-readable, sanitized failures and bounded request timeouts.
    from contextlib import contextmanager
    import inspect
    import scaletrain_cli.gateways.live_train_gateway as gateway_module
    from train_client import ApiClient, Configuration
    from train_client.api.default_api import DefaultApi
    original_call = ApiClient.call_api
    signature = inspect.signature(original_call)
    def bounded_call(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        if bound.arguments.get("_request_timeout") is None:
            bound.arguments["_request_timeout"] = (10, 40)
        return original_call(*bound.args, **bound.kwargs)
    ApiClient.call_api = bounded_call
    @contextmanager
    def private_api(host):
        with ApiClient(configuration=Configuration(host=host)) as client:
            yield DefaultApi(client)
    gateway_module.scaletrain_api_instance = private_api
    from scaletrain_cli.gateways.live_train_gateway import LiveTrainGateway
    from scaletrain_cli.repositories.filesystem_configuration_repository import FilesystemConfigurationRepository
    gateway = LiveTrainGateway(FilesystemConfigurationRepository.read().get_train_service_host_url())
    if args.command in {"validate", "submit"}:
        data = json.loads(Path(args.job_file).read_text())
        model = validate(data)
        if args.command == "validate":
            result = {"valid": True, "job_name": data["job_name"], "nodes": data["num_instances"],
                      "maximum_node_hours": data["num_instances"] * data["active_deadline_seconds"] / 3600}
        else:
            receipt = Path(args.receipt)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            # Persistent intent blocks duplicate creation after an ambiguous network response.
            if receipt.exists():
                raise ValueError("Submission receipt already exists; reconcile it using list/get before retrying.")
            jobs = gateway.list_jobs(user_id=args.user_id, limit=100)
            matches = [j for j in jobs if j.name == data["job_name"]]
            if matches:
                raise ValueError("Job name already exists; monitor it instead of submitting a duplicate.")
            with receipt.open("x") as f:
                json.dump({"state": "submission_pending", "job_name": data["job_name"], "user_id": args.user_id}, f)
            job = gateway.train(model, creator_user_id=args.user_id, tags={"tool": "strl"})
            result = summary(job)
            receipt.write_text(json.dumps(result, indent=2, default=str))
    elif args.command == "list":
        result = [summary(j) for j in gateway.list_jobs(user_id=args.user_id, limit=args.limit)]
    elif args.command == "get":
        result = summary(gateway.get_job(args.job_id))
    elif args.command == "progress":
        result = progress_summary(gateway.list_job_progress_entries(args.job_id))
    elif args.command == "queue":
        result = [summary(j) for j in gateway.get_queue(args.instance_type, limit=args.limit)]
    else:
        job = gateway.get_job(args.job_id)
        if job.created_by != args.user_id:
            raise ValueError("Cancellation owner mismatch.")
        result = summary(job) if str(job.status) in TERMINAL else {"cancel_requested": bool(gateway.cancel_job(args.job_id))}
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # API exceptions may contain credential-bearing headers or payloads.
        safe = str(exc) if isinstance(exc, ValueError) else "Check CLI configuration, authentication, VPN, and API access."
        print(json.dumps({"error": type(exc).__name__, "detail": safe}), file=sys.stderr)
        sys.exit(1)
