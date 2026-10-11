"""strl: one CLI for the whole experiment lifecycle. Run from the repo root with the local control venv (.venv).

  plan      <exp> [--against <exp>] [--set K=V] [--network network.json]
                                           resolve, validate, derive infra + layout, estimate time/cost/memory, print warnings
  fetch     <exp>                          download the pinned model + router wheel, then prepare the environment's data
  prepare   <exp>                          (re)build the environment's data only (coding: validates references in Modal)
  executor  deploy|check                   the Modal code executor used by coding rewards
  overlay                                  build assets/overlay (Modal/Harbor packages for the training image)
  capacity  [--gpus-per-node 8,4,2,1]      read-only ScaleTrain/EKS capacity by team and borrowing route
  stage     <exp> --attempt A --mode dry|canary|train [--set K=V]  upload assets + render job.json (never submits)
  check-template <exp>                     does the model's chat template keep step-wise trajectories mergeable (SkyRL prefix merge)?
  stage-model <exp> --attempt A            render a CPU job that downloads the model in a pod and uploads it to S3 (large models)
  submit    <job.json> [--user-id U]       submit once; receipt written before the POST
  launch    <exp> --attempt A --mode M [--set K=V] [--routes fb,rb,ro,fo] [--race-shape "K=V ..."] [--max-wait-hours H] [--watch] [--relaunch-on-preempt N]
                                           stage + submit one job per capacity route, keep the first that starts, cancel the rest (USE THIS)
  status    <job_id> | --list              job state
  monitor   <exp> --attempt A [--job-id J] live status line + alerts until the attempt finishes
  ledger    <exp> --attempt A --job-id J   record a terminal attempt's allocated interval in the program budget ledger
  verify    <exp> --attempt A              independent re-scoring of the final evaluation from S3
  compare   <exp>:<attempt> ...            paired comparison of finished attempts on the same held-out ids
  report    <exp>:<attempt> ... [--no-wandb]  evidence/results/<group>-report.md (headline, per-domain, curves, infra) + a W&B report run
  eval-llm  <exp> --model M                reference eval of an external LLM with the training prompts/harness
  critic    <exp> [--n 5]                  run the LiteLLM critic on sample failures (no GPUs)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = os.environ.get('STRL_SCALETRAIN_USER', 'mohamed.elfeki')


def _load(path: str, overrides: list[str] | None = None) -> dict:
    from .experiment import ExperimentError, load
    try:
        return load(path, overrides=overrides)
    except ExperimentError as error:
        print(f'experiment {path} is invalid:\n{error}', file=sys.stderr)
        raise SystemExit(2)


_AWS_OK: set[str] = set()


def require_aws(cfg: dict) -> None:
    """Fail early with the fix instead of a botocore traceback mid-stage: every S3-touching command needs a live SSO session for the
    experiment's storage profile (it expires; renewing it is the user's step: `aws sso login --profile <profile>`)."""
    profile = cfg['storage']['profile']
    if profile in _AWS_OK or os.environ.get('STRL_SKIP_AWS_CHECK'):
        return
    try:
        import boto3
        boto3.Session(profile_name=profile).client('sts', region_name=cfg['storage'].get('region')).get_caller_identity()
    except Exception as error:   # TokenRetrievalError (expired SSO), NoCredentialsError, ProfileNotFound ...
        print(f'AWS credentials for profile {profile!r} are not usable ({type(error).__name__}: {str(error)[:160]}).\n'
              f'Nothing was staged or submitted. Ask the user to run:  aws sso login --profile {profile}\n'
              '(an agent must not log in on the user\'s behalf), then re-run this command.', file=sys.stderr)
        raise SystemExit(2)
    _AWS_OK.add(profile)


def readiness_warnings(cfg: dict) -> list[str]:
    """What `plan` cannot derive from the YAML alone: is the data prepared for THIS selection, and does the budget ledger exist?"""
    out = []
    from .tasks import get_environment
    try:
        get_environment(cfg['environment']['name']).assert_manifest(cfg)
    except Exception as error:   # missing dir, fewer prepared tasks than configured, other filters/seed/revision
        out.append(f"DATA: {cfg['run'].get('data_dir')} is not prepared for this selection ({' '.join(str(error).split())[:240]}): run "
                   '`strl prepare <exp>` (or `strl fetch`). Lowering train/test counts reuses a prepared dir; more tasks, other filters or another seed need a prepare.')
    ledger = cfg['budget'].get('ledger')
    b = cfg['budget']
    if int(b['canary_max_seconds']) + 600 > 3600:
        out.append(f"CANARY: budget.canary_max_seconds={b['canary_max_seconds']} + 600 s exceeds the 3600 s canary job cap: the canary job gets "
                   '3600 s (a slow step-0 eval + 2 steps of agentic trials may not fit: lower the per-domain caps or the step deadline for it)')
    if ledger and Path(ledger).is_file():
        from .budget import node_fraction, remaining_seconds, reserved_seconds
        from .stage import CANARY_OVERLAY, canary_overlay_for
        data = json.loads(Path(ledger).read_text())
        left = remaining_seconds(data['attempts'], data['total_seconds']) - reserved_seconds(data)
        charge = node_fraction(cfg)
        canary_charge = 1 / 8 if canary_overlay_for(cfg) == CANARY_OVERLAY else charge
        # what stage refuses below (strl/stage.py) and what it books for the FULL attempt (attempt seconds + 420 s pod-start slack)
        need_train = int((b['final_eval_reserve_seconds'] + 901) * charge)
        need_canary = int(900 * canary_charge)
        full_train = int((b['attempt_max_seconds'] + 420) * charge)
        full_canary = int((min(3600, int(b['canary_max_seconds']) + 600) + 420) * canary_charge)
        if left < min(need_train, need_canary):
            out.append(f'BUDGET: {left} node-s available in {ledger}: no GPU mode fits (a train attempt needs > {need_train}, a canary >= '
                       f'{need_canary}); dry runs still work. More GPU budget needs the user\'s authorization and a new ledger.')
        elif left < need_train:
            out.append(f'BUDGET: {left} node-s available in {ledger}: a canary fits (it books {min(full_canary, left)}), a train attempt does not '
                       f'(needs > {need_train}).')
        elif charge and left < b['attempt_max_seconds'] * charge:   # strl/stage.py: attempt seconds = min(left / charge, attempt_max)
            out.append(f'BUDGET: {left} node-s available in {ledger}: stage would CLAMP a train attempt to {int(left / charge)} s instead of '
                       f'budget.attempt_max_seconds={b["attempt_max_seconds"]} (fewer steps than the estimate above; lower max_steps or ask '
                       'for more budget).')
        elif left < full_train:
            out.append(f'BUDGET: {left} node-s available in {ledger}: a full train attempt runs but books {full_train} (its 420 s pod-start '
                       'slack included), more than is left: a concurrent launch on this ledger would be refused until it is recorded.')
        if left >= need_train and left < full_train + full_canary:
            out.append(f'BUDGET: a separate canary ({full_canary}) plus a full train attempt ({full_train}) do not both fit in {left}: run a short '
                       'train attempt (max_steps 2) as the canary, or size attempt_max_seconds so both fit (launch-and-capacity.md "Sizing").')
    if ledger and not Path(ledger).is_file():
        out.append(f'LEDGER: budget.ledger {ledger} does not exist, so stage will refuse. A new GPU program needs the user\'s authorization; then create '
                   'the file (references/launch-and-capacity.md, "New program, new budget") and set budget.total_seconds to match.')
    return out


def cmd_plan(a):
    from .derive import plan
    from .experiment import diff
    cfg = _load(a.experiment, a.set)
    try:
        network = json.loads(Path(a.network).read_text()) if a.network else None
        if network is None and int(cfg['compute']['nodes']) > 1:   # the last measured multi-node probe (strl/profiles.json networks)
            measured = json.loads((ROOT / 'strl/profiles.json').read_text()).get('networks') or []
            network = dict(measured[-1]) if measured else None
            if network:   # another job's probe: an assumption, labelled as such (pass --network <attempt>/network.json for your own)
                network['source'] = f"ASSUMED from {network.get('source')}; --network <your probe> to replace"
        report = plan(cfg, network)
    except RuntimeError as error:   # model detection (gated / offline config, contradicting profile, unmapped VLM wrapper)
        print(f'experiment {a.experiment} cannot be planned:\n- {error}', file=sys.stderr)
        raise SystemExit(2)
    if a.against:
        other = _load(a.against)
        rows = [{'key': k, 'from': x, 'to': y} for k, x, y in diff(other, cfg)]
        from .tasks import get_environment
        derived = lambda c: get_environment(c['environment']['name']).skyrl_overrides(c, 'train').get('environment.env_class') or 'generator'
        if derived(other) != derived(cfg):   # e.g. a custom reward term moves gsm8k from SkyRL's built-in env to strl_env (proven equivalent)
            rows.append({'key': 'derived: environment.env_class', 'from': derived(other), 'to': derived(cfg)})
        report['diff_vs'] = {a.against: rows}
        ledger = Path(cfg['budget'].get('ledger') or '')
        if ledger.is_file() and other['budget'].get('ledger') == cfg['budget'].get('ledger'):   # an A/B: both arms draw on one ledger
            from .budget import node_fraction, remaining_seconds, reserved_seconds
            data = json.loads(ledger.read_text())
            left = remaining_seconds(data['attempts'], data['total_seconds']) - reserved_seconds(data)
            both = sum(int(node_fraction(c) * (c['budget']['attempt_max_seconds'] + 420)) for c in (cfg, other))
            if left < both:
                report['warnings'].append(f'BUDGET: the two arms book {both} node-s together but {ledger} has {left}: launched together, the '
                                          'second would be refused (or clamped to fewer steps: an unfair A/B). Ask for budget or shorten both.')
    report['warnings'] += readiness_warnings(cfg)
    if a.json:
        print(json.dumps(report, indent=2, default=str))
        return
    print(f"experiment {report['experiment']}  sha {report['sha']}\n  chain: {' -> '.join(Path(c).name for c in report['chain'])}")
    for k, v in report['axes'].items():
        print(f'  {k:12s} {v}')
    d, e = report['derived'], report['estimates']
    print(f"  placement    {d['world_size']} GPU(s), {d['num_engines']} vLLM engine(s), {d['rollouts_per_step']} rollouts/step "
          f"({d['rollouts_per_gpu']}/GPU, {d['micro_batches_per_gpu']} micro-batches), max_model_len {d['max_model_len']}, env_class {d['skyrl_env_class']}")
    print(f"  estimates    step ~{e['step_s']}s (generate {e['generate_s']} + train {e['train_s']} + other {e['other_s']}), ~{e['steps_in_attempt']} steps/attempt, "
          f"{e['node_hours']:.2f} node-h; memory train {e['memory']['training_phase_gb']} GB / rollout {e['memory']['rollout_phase_gb']} GB "
          + (f"[measured peak {e['memory']['measured_peak_gb']} GB in {e['memory']['measured_in']}] " if e['memory'].get('measured_peak_gb') else '')
          + f"(fits={e['memory']['fits']}); from profile {e['profile']}"
          + ("" if e.get('calibrated') else f" (UNCALIBRATED: {e.get('uncalibrated_because') or 'different environment/model'})"))
    lay, mem = report['layout'], e['memory']
    print(f"  layout       {lay['mode']} | FSDP group {lay['fsdp_group']} x {lay['replicas']} replica(s), cpu_offload={lay['cpu_offload']}"
          + (f" (host RAM {mem['host_offload_gb_per_gpu']} GB/GPU)" if mem.get('host_offload_gb_per_gpu') else '') + " | "
          f"vLLM TP{lay['tp']} x {lay['engines']} engine(s) | per-GPU train {lay['train_gb_per_gpu']} GB, rollout {lay['rollout_gb_per_gpu']} GB "
          f"(weights {mem['engine_weights_gb']}, KV room {mem['kv_room_gb']})"
          + (f", LoRA {mem['lora_params_m']}M params" if mem['lora_params_m'] else '') + f"; recommended: {lay['recommended']['reason']}")
    print(f"  monitoring   {report['monitoring']['wandb']}  critic={report['monitoring']['critic']}")
    ledger_path = Path(cfg['budget'].get('ledger') or '')
    if ledger_path.is_file():   # the program budget the next stage would draw from (node-seconds)
        from .budget import remaining_seconds
        ledger = json.loads(ledger_path.read_text())
        from .budget import reserved_seconds
        held = reserved_seconds(ledger)
        left = remaining_seconds(ledger['attempts'], ledger['total_seconds'])
        from .budget import node_fraction
        from .stage import CANARY_OVERLAY, canary_overlay_for
        train = int(node_fraction(cfg) * (cfg['budget']['attempt_max_seconds'] + 420))   # stage books the 420 s pod-start slack too
        small = canary_overlay_for(cfg) == CANARY_OVERLAY
        canary = int((min(3600, cfg['budget']['canary_max_seconds'] + 600) + 420) * (1 / 8 if small else node_fraction(cfg)))
        shape = '1-GPU' if small else '{}x{}-GPU'.format(cfg['compute']['nodes'], cfg['compute']['gpus_per_node'])
        print(f"  budget       {ledger_path}: {left} of {ledger['total_seconds']} node-s left"
              + (f" ({held} held by {len(ledger.get('reservations') or {})} staged launch(es))" if held else '')
              + f"; a train attempt reserves up to {train} node-s, a {shape} canary up to {canary} ({len(ledger['attempts'])} attempts recorded)")
    for w in report['warnings']:
        print(f'  ! {w}')
    for k, rows in report.get('diff_vs', {}).items():
        print(f'  diff vs {k}:')
        for row in rows:
            print(f"     {row['key']}: {row['from']!r} -> {row['to']!r}")


def cmd_fetch(a):
    cfg = _load(a.experiment)
    from huggingface_hub import snapshot_download
    from .models import snapshot_identity, write_identity
    if snapshot_identity(cfg['model'])[0] == 'mismatch':
        raise SystemExit(snapshot_identity(cfg['model'])[1])
    # top-level files only, exactly what stage uploads and stage-model copies: repos often ship onnx/, runs/ or other subfolders many times
    # larger than the weights (SmolLM2-1.7B: 19 GB of ONNX next to 3.4 GB of safetensors)
    snapshot_download(cfg['model']['id'], revision=cfg['model']['revision'], local_dir=cfg['model']['local_path'],
                      ignore_patterns=['*/*', 'README.md', 'LICENSE', '.gitattributes'])
    write_identity(cfg['model'])
    from .runtime import ROUTER_SHA256, ROUTER_WHEEL
    import hashlib
    runtime = ROOT / 'assets/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    wheel = runtime / ROUTER_WHEEL
    if not wheel.exists():
        import urllib.request
        urllib.request.urlretrieve('https://github.com/SumanthRH/router/releases/download/0.1.14.post1/' + ROUTER_WHEEL, wheel)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != ROUTER_SHA256:
        raise SystemExit('vllm-router wheel hash mismatch')
    from .tasks import get_environment
    try:
        get_environment(cfg['environment']['name']).assert_manifest(cfg)
        print(f"data already prepared and verified at {cfg['run']['data_dir']}")
    except Exception:
        cmd_prepare(a)


def cmd_prepare(a):
    cfg = _load(a.experiment)
    from .tasks import get_environment
    env = get_environment(cfg['environment']['name'])
    kwargs = {}
    if env.needs_modal and env.uses_coding_executor:
        from transformers import AutoTokenizer
        from .services.modal import local_profile
        kwargs['tokenizer'] = AutoTokenizer.from_pretrained(cfg['model']['local_path'])
        with local_profile(cfg['services']['modal'].get('environment', 'main')):
            from .tasks.coding import executor
            print(json.dumps(executor.ensure_deployed()))
            manifest = env.prepare_data(cfg, **kwargs)
    elif env.needs_modal:   # e.g. mimo: the exporter captures per-task state (base refs, pristine binaries) in Modal sandboxes
        from .services.modal import local_profile
        with local_profile(cfg['services']['modal'].get('environment', 'main')):
            manifest = env.prepare_data(cfg)
    else:
        manifest = env.prepare_data(cfg)
    print(json.dumps({k: manifest.get(k) for k in ('counts', 'files', 'prompt_tokens', 'executor_version') if k in manifest} or
                     {'files': manifest.get('files')}, indent=1, default=str))
    if manifest.get('dropped'):
        print(f"dropped {len(manifest['dropped'])} task(s): " + json.dumps(dict(list(manifest['dropped'].items())[:10]), indent=1))


def cmd_executor(a):
    from .services.modal import local_profile
    from .tasks.coding import executor
    with local_profile(a.environment):
        print(json.dumps(executor.deploy() if a.action == 'deploy' else executor.ensure_deployed(auto_deploy=False)))


def cmd_overlay(a):
    subprocess.run([sys.executable, str(ROOT / 'tools/build_overlay.py'), *(['--force'] if a.force else [])], check=True)


def cmd_capacity(a):
    subprocess.run([sys.executable, str(ROOT / 'tools/capacity.py'), '--nodes', str(a.nodes), '--gpus-per-node', a.gpus_per_node,
                    '--output', a.output], check=True, env={**os.environ, 'AWS_PROFILE': os.environ.get('AWS_PROFILE', 'ml-admin')})


def cmd_stage(a):
    from .stage import stage
    require_aws(_load(a.experiment, a.set))
    receipt = stage(a.experiment, attempt=a.attempt, mode=a.mode, team=a.team, allow_borrowing=a.allow_borrowing, ledger=a.ledger,
                    output=a.output or f'evidence/{a.attempt}/job.json', overrides=[*(a.set or []), *(['compute.allow_borrowing=false'] if a.no_borrowing else [])])
    Path(f'evidence/{a.attempt}').mkdir(parents=True, exist_ok=True)
    Path(f'evidence/{a.attempt}/stage.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=1))
    print(f"\nnext: python -m strl submit {receipt['job_file']}")


def cmd_check_template(a):
    from .models import template_check
    cfg = _load(a.experiment, a.set)
    try:
        result = template_check.check(cfg, online=True)
    except RuntimeError as error:
        print(f'cannot check {a.experiment}:\n- {error}', file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result or {'ok': None, 'detail': 'tokenizer unavailable'}, indent=1))


def cmd_stage_model(a):
    from .stage import stage_model
    require_aws(_load(a.experiment, a.set))
    receipt = stage_model(a.experiment, attempt=a.attempt, team=a.team, output=a.output or f'evidence/{a.attempt}/job.json', overrides=a.set)
    Path(f'evidence/{a.attempt}').mkdir(parents=True, exist_ok=True)
    Path(f'evidence/{a.attempt}/stage.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=1))
    print(f"\nnext: python -m strl submit {receipt['job_file']}  (CPU; then stage the experiment: it reads {receipt['manifest']})")


def _scaletrain(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / 'tools/scaletrain.py'), *args], capture_output=True, text=True)


def cmd_submit(a):
    job = json.loads(Path(a.job).read_text())
    receipt = a.receipt or str(Path(a.job).parent / 'submission.json')
    check = _scaletrain('validate', a.job)
    if check.returncode:
        raise SystemExit(check.stderr or check.stdout)
    out = _scaletrain('submit', a.job, '--user-id', a.user_id, '--receipt', receipt)
    if out.returncode:  # never resubmit blindly: reconcile by exact job name
        listed = _scaletrain('list', '--user-id', a.user_id, '--limit', '20')
        listing = json.loads(listed.stdout or '[]') if not listed.returncode else None
        match = [j for j in listing or [] if j['name'] == job['job_name']]
        print(json.dumps({'submit_error': (out.stderr or out.stdout).strip()[-300:], 'reconciled_by_name': match[:1],
                          'listing_ok': listing is not None}, indent=1))
        receipt_file = Path(a.job).parent / 'stage.json'
        definitely_absent = listing is not None and not match   # an ambiguous failure keeps the booking (the job may be live)
        if definitely_absent and receipt_file.is_file() and json.loads(receipt_file.read_text()).get('ledger'):   # rejected: free its booking
            from .stage import release_reservation
            release_reservation(ROOT / json.loads(receipt_file.read_text())['ledger'], None, job['job_name'])
        return
    print(out.stdout.strip())


def cmd_status(a):
    out = _scaletrain('list', '--user-id', a.user_id, '--limit', '20') if a.list else _scaletrain('get', a.job_id)
    print(out.stdout.strip() or out.stderr.strip())


def cmd_monitor(a):
    from .monitor.live import watch
    require_aws(_load(a.experiment))
    watch(_load(a.experiment), a.attempt, a.job_id, interval=a.interval, once=a.once)


def worker_started(cfg: dict, attempt: str, s3=None) -> bool:
    """True if a worker ever ran in this attempt's pod (it writes artifacts/worker-started.json first thing). Fails closed: any S3 error other
    than "not found" raises, so a claim of "never started" is only ever made on positive evidence."""
    import boto3
    from botocore.exceptions import ClientError
    st = cfg['storage']
    s3 = s3 or boto3.Session(profile_name=st['profile']).client('s3')
    try:
        s3.head_object(Bucket=st['bucket'], Key=f"{st['prefix']}/attempts/{attempt}/artifacts/worker-started.json")
        return True
    except ClientError as error:
        if error.response.get('Error', {}).get('Code') in {'404', 'NoSuchKey', 'NotFound'}:
            return False
        raise


POD_START_SLACK_SECONDS = 420   # image pull, container start and the in-pod payload/model bootstrap (S3 fetch + sha256) happen before the worker writes budget.json


def allocation_start(cfg: dict, attempt: str, s3=None) -> float | None:
    """Epoch seconds at which this attempt's pods were running: the worker's write-once ``budget.json`` ``started_at`` (the FIRST incarnation
    when a preempted job was requeued) minus the pod-start slack. None when no worker ever ran. ScaleTrain reports no allocation times, and
    charging ``created_at`` made a 2 x 8 job that waited 3.5 h in the queue cost more than the whole 6 node-hour program."""
    import boto3
    from botocore.exceptions import ClientError
    st = cfg['storage']
    s3 = s3 or boto3.Session(profile_name=st['profile']).client('s3')
    try:
        body = s3.get_object(Bucket=st['bucket'], Key=f"{st['prefix']}/attempts/{attempt}/artifacts/budget.json")['Body'].read()
    except ClientError as error:
        if error.response.get('Error', {}).get('Code') in {'404', 'NoSuchKey', 'NotFound'}:
            return None
        raise
    return float(json.loads(body)['started_at']) - POD_START_SLACK_SECONDS


def record_attempt(experiment: str, attempt: str, job_id: str, *, terminal_at: str | None = None, never_started: bool = False,
                   ledger: str | None = None, job: dict | None = None, overrides: list[str] | None = None,
                   allocated_floor: float | None = None) -> dict:
    """Append one terminal attempt to the program ledger. Returns {'recorded', 'remaining_node_seconds'}; raises ValueError when it must not.

    Charged from the pod start (budget.json started_at - slack) when a worker ran; otherwise from ``allocated_floor`` (the launcher's first
    observation of the job leaving the queue, minus one poll) when known, else conservatively from created_at. ``never_started`` (positive
    evidence that no pod ran) records it uncharged for ANY terminal status. ``overrides`` are the launch's --set values (ledger, storage).
    The read-modify-write is serialized with a file lock and an atomic replace: concurrent launchers share one ledger."""
    cfg = _load(experiment, overrides)
    ledger_path = Path(ledger) if ledger else ROOT / cfg['budget']['ledger']
    job = job or json.loads(_scaletrain('get', job_id).stdout)
    if job['status'] not in {'COMPLETED', 'FAILED', 'CANCELED', 'CANCELLED'}:
        raise ValueError(f'job is {job["status"]}; record it only after a terminal state')
    stage_receipt = json.loads(Path(f'evidence/{attempt}/stage.json').read_text())
    created = str(job['created_at']).replace(' ', 'T')[:19] + 'Z'
    entry = {'job_id': job_id, 'attempt': attempt, 'status': job['status'], 'allocated_at': created,
             'terminal_at': terminal_at or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu_fraction': stage_receipt['gpu_fraction'],
             'nodes': int(stage_receipt.get('nodes', 1)),
             'evidence': 'conservative: job created_at to first terminal observation (includes queue/pod start)'}
    created_s = datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp()
    if never_started:   # no pod, no GPU time -> not charged (budget.remaining_seconds skips an unallocated terminal job)
        if worker_started(cfg, attempt) or allocation_start(cfg, attempt) is not None:   # budget.json is written before rendezvous: pods were up
            raise ValueError(f'{attempt} has worker-started.json or budget.json: a pod ran, so it is charged (omit --never-started)')
        entry.update(allocated_at=None, evidence=f"{job['status']} without a pod: no worker-started.json / budget.json in the attempt artifacts (not charged)")
    else:
        started = allocation_start(cfg, attempt)
        if started is not None:   # a worker ran: charge the time the pods were up, never the time the job waited in the queue
            entry.update(allocated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(max(created_s, started))),
                         evidence=f'worker budget.json started_at - {POD_START_SLACK_SECONDS} s (pods up) to the terminal observation; queue time (created_at {created}) is not charged')
        elif allocated_floor is not None:   # allocated but no worker trace yet (a cancelled loser, a bootstrap failure)
            entry.update(allocated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(max(created_s, allocated_floor))),
                         evidence='no worker trace: charged from the launcher\'s first observation of the job leaving the queue (minus one poll) '
                                  f'to the terminal observation; queue time (created_at {created}) is not charged')
    from .budget import locked_ledger
    with locked_ledger(ledger_path) as data:
        if any(x['job_id'] == job_id for x in data['attempts']):
            raise ValueError(f'{job_id} already recorded')
        data['attempts'].append(entry)
        from .budget import release_attempt
        release_attempt(data, attempt)   # release this attempt's share of its launch's reservation
    from .budget import remaining_seconds
    return {'recorded': entry, 'remaining_node_seconds': remaining_seconds(data['attempts'], data['total_seconds'])}


def cmd_ledger(a):
    if a.release:   # a staged attempt that will never be submitted (or was rejected): free its budget booking
        from .stage import release_reservation
        ledger = Path(a.ledger) if a.ledger else ROOT / _load(a.experiment, a.set)['budget']['ledger']
        print(json.dumps({'released': release_reservation(ledger, None, a.release), 'attempt': a.release}))
        return
    if not (a.attempt and a.job_id):
        raise SystemExit('strl ledger needs --attempt and --job-id to record a terminal attempt (or --release ATTEMPT)')
    try:
        out = record_attempt(a.experiment, a.attempt, a.job_id, terminal_at=a.terminal_at, never_started=a.never_started, ledger=a.ledger,
                             overrides=a.set)
    except ValueError as error:
        raise SystemExit(str(error))
    print(json.dumps(out, indent=1))


def cmd_launch(a):
    from .launch import main_launch
    raise SystemExit(main_launch(a))


def cmd_verify(a):
    from .verify_results import verify
    local = Path('evidence') / a.attempt
    if not any((local / name).is_file() for name in ('result.json', 'base.jsonl', 'completion.json')):   # already-downloaded evidence works offline
        require_aws(_load(a.experiment))
    try:
        out = verify(_load(a.experiment), a.attempt)
    except (RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f'cannot verify {a.attempt}: {error}')
    print(json.dumps(out, indent=1, default=str))


def cmd_compare(a):
    from .monitor.compare import compare, markdown
    report = compare(a.runs)
    Path('evidence/compare').mkdir(parents=True, exist_ok=True)
    Path('evidence/compare/report.json').write_text(json.dumps(report, indent=2, default=str))
    print(markdown(report))


def cmd_report(a):
    from .monitor.report import write_reports
    for entry in write_reports(a.runs, wandb=not a.no_wandb):
        print(json.dumps(entry, default=str))
        print(Path(entry['path']).read_text())


def cmd_eval_llm(a):
    from . import evals
    cfg = _load(a.experiment)
    project = ((cfg.get('services') or {}).get('litellm') or {}).get('project_id', '696011502a26dc4a64b24f55')
    if cfg['environment']['name'] == 'mimo':   # the ceiling of a mimo selection: --split test|canary, --limit = tasks per domain
        print(json.dumps(evals.mimo(cfg, a.split if a.split != 'validation' else 'test', a.limit, a.model, a.max_completion_tokens, project), indent=1))
        return
    fn = evals.coding_harbor if cfg['environment']['name'] == 'coding_harbor' or a.harbor else evals.coding_exec
    if fn is evals.coding_exec:
        from .services.modal import local_profile
        with local_profile(cfg['services']['modal'].get('environment', 'main')):
            print(json.dumps(fn(cfg, a.split, a.limit, a.model, a.max_completion_tokens, project), indent=1))
    else:
        print(json.dumps(fn(cfg, a.split, a.limit, a.model, a.max_completion_tokens, project), indent=1))


def cmd_critic(a):
    from .monitor import critic
    from .tasks.coding import data as coding_data
    cfg = _load(a.experiment)
    critic.configure({**cfg['tracking'].get('critic', {}), 'enabled': True})
    tasks = coding_data.load_tasks(cfg, 'canary')[:a.n] if cfg['environment']['name'].startswith('coding') else []
    rows = []
    for task in tasks:
        broken = coding_data.stub(task.reference)
        rows.append(critic.analyse({'source_step': 0, 'idx': 0, 'prompt': task.prompt, 'completion': f'```python\n{broken}\n```',
                                    'env_failure': 'wrong_answer'}))
    print(json.dumps([{k: r[k] for k in ('critic_mode', 'rationale', 'fix_hint', '_cost')} for r in rows], indent=1))
    print(json.dumps({'calls': len(rows), 'cost_usd': round(sum(r['_cost'] for r in rows), 5)}))


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m strl', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan'); s.add_argument('experiment'); s.add_argument('--against'); s.add_argument('--json', action='store_true')
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='dotted experiment override, YAML value (repeatable)')
    s.add_argument('--network', help="a probe result (an attempt's artifacts/network.json) for the layout recommendation")
    s = sub.add_parser('fetch'); s.add_argument('experiment')
    s = sub.add_parser('prepare'); s.add_argument('experiment')
    s = sub.add_parser('executor'); s.add_argument('action', choices=['deploy', 'check']); s.add_argument('--environment', default='main')
    s = sub.add_parser('overlay'); s.add_argument('--force', action='store_true')
    s = sub.add_parser('capacity'); s.add_argument('--nodes', type=int, default=1); s.add_argument('--gpus-per-node', default='8,4,2,1')
    s.add_argument('--output', default='evidence/capacity.json')
    s = sub.add_parser('launch'); s.add_argument('experiment'); s.add_argument('--attempt', required=True)
    s.add_argument('--mode', choices=['dry', 'canary', 'train'], required=True); s.add_argument('--set', action='append', default=[])
    s.add_argument('--routes', default='all', help='comma list of fb (foundation+borrowing), rb (research+borrowing), ro (research own quota), fo (foundation own quota)')
    s.add_argument('--poll', type=float, default=15.0); s.add_argument('--max-wait-hours', type=float, default=None)
    s.add_argument('--watch', action='store_true', help='follow the winner to its end, record it, detect preemption')
    s.add_argument('--relaunch-on-preempt', type=int, default=0); s.add_argument('--user-id', default=USER)
    s.add_argument('--fallback-set', action='append', default=[], help='after --max-wait-hours with no capacity, race again with these overrides (e.g. compute.nodes=1)')
    s.add_argument('--race-shape', action='append', default=[], metavar='"K=V [K=V ...]"',
                   help='an alternative shape raced IN ADDITION to the experiment\'s own (e.g. "compute.nodes=1"); its routes are submitted once the '
                        'own shape has waited --race-shape-after-minutes; the first job to start wins (attempt <A>-s<k>-<route>)')
    s.add_argument('--race-shape-after-minutes', type=float, default=10.0)
    s = sub.add_parser('stage'); s.add_argument('experiment'); s.add_argument('--attempt', required=True)
    s.add_argument('--mode', choices=['dry', 'canary', 'train'], required=True); s.add_argument('--team'); s.add_argument('--allow-borrowing', action='store_true'); s.add_argument('--no-borrowing', action='store_true', help='own-quota route only (not reclaimable): sets compute.allow_borrowing=false')
    s.add_argument('--output'); s.add_argument('--ledger', help='program budget ledger (default: budget.ledger of the experiment)')
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='dotted experiment override, YAML value (repeatable)')
    s = sub.add_parser('check-template'); s.add_argument('experiment'); s.add_argument('--set', action='append', default=[])
    s = sub.add_parser('stage-model'); s.add_argument('experiment'); s.add_argument('--attempt', required=True); s.add_argument('--team')
    s.add_argument('--output'); s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    s = sub.add_parser('submit'); s.add_argument('job'); s.add_argument('--user-id', default=USER); s.add_argument('--receipt')
    s = sub.add_parser('status'); s.add_argument('job_id', nargs='?'); s.add_argument('--list', action='store_true'); s.add_argument('--user-id', default=USER)
    s = sub.add_parser('monitor'); s.add_argument('experiment'); s.add_argument('--attempt', required=True); s.add_argument('--job-id')
    s.add_argument('--interval', type=float, default=60); s.add_argument('--once', action='store_true')
    s = sub.add_parser('ledger'); s.add_argument('experiment'); s.add_argument('--attempt'); s.add_argument('--job-id')
    s.add_argument('--release', metavar='ATTEMPT', help='drop a staged-but-never-submitted attempt from its budget reservation')
    s.add_argument('--terminal-at', help='ISO time of the terminal observation (default: now)')
    s.add_argument('--never-started', action='store_true', help='a terminal job whose pod never ran (verified: no worker-started.json / budget.json): not charged')
    s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help="the launch's --set overrides (ledger, storage)")
    s.add_argument('--ledger', help='program budget ledger (default: budget.ledger of the experiment)')
    s = sub.add_parser('verify'); s.add_argument('experiment'); s.add_argument('--attempt', required=True)
    s = sub.add_parser('compare'); s.add_argument('runs', nargs='+', help='experiment.yaml:attempt')
    s = sub.add_parser('report'); s.add_argument('runs', nargs='+', help='experiment.yaml:attempt'); s.add_argument('--no-wandb', action='store_true')
    s = sub.add_parser('eval-llm'); s.add_argument('experiment'); s.add_argument('--model', default='openai/gpt-6-luna')
    s.add_argument('--split', default='validation'); s.add_argument('--limit', type=int, default=20); s.add_argument('--max-completion-tokens', type=int, default=4096)
    s.add_argument('--harbor', action='store_true', help='use the Harbor/Terminus-2 agent even for coding_exec tasks')
    s = sub.add_parser('critic'); s.add_argument('experiment'); s.add_argument('--n', type=int, default=5)
    a = p.parse_args(argv)
    os.chdir(ROOT)
    {'plan': cmd_plan, 'fetch': cmd_fetch, 'prepare': cmd_prepare, 'executor': cmd_executor, 'overlay': cmd_overlay, 'capacity': cmd_capacity,
     'stage': cmd_stage, 'stage-model': cmd_stage_model, 'check-template': cmd_check_template, 'submit': cmd_submit, 'status': cmd_status, 'monitor': cmd_monitor, 'ledger': cmd_ledger, 'verify': cmd_verify,
     'compare': cmd_compare, 'report': cmd_report, 'eval-llm': cmd_eval_llm, 'critic': cmd_critic, 'launch': cmd_launch}[a.cmd](a)


if __name__ == '__main__':
    main()
