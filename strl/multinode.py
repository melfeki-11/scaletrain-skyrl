"""Multi-node rendezvous, Ray cluster formation and liveness (ported from the verified two-node qwen35-grpo run, q35/).

rank = JOB_COMPLETION_INDEX, nodes = NUM_INSTANCES. The head publishes its routable job-interface address to
<artifact_uri>/leader-network.json (never leader DNS: it depends on head container readiness); peers poll that record, pick a
local address of the same family, and join Ray with bounded retries. Liveness, both ways:
  head  heartbeat <artifact_uri>/head-heartbeat.json every 60 s; aborts training when fewer than N Ray nodes are alive
  peer  heartbeat <artifact_uri>/peer-<rank>-heartbeat.json every 60 s; exits 0 once completion.json exists and nonzero when
        every head record (head-heartbeat.json, heartbeat.json, leader-network.json) is older than HEAD_STALE_SECONDS
Single-node jobs never import this module.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from typing import Any, Callable

from .network import endpoint, interface_addresses, ray_start_args, routable, select_addresses
from .storage import split_uri

LEADER = 'leader-network.json'
HEAD_HEARTBEAT = 'head-heartbeat.json'
HEAD_RECORDS = (HEAD_HEARTBEAT, 'heartbeat.json', LEADER)
HEARTBEAT_SECONDS = 60
HEAD_STALE_SECONDS = 300
RAY = '/opt/skyrl-venv/bin/ray'


class HeadStale(RuntimeError):
    pass


def rank_and_world(environ=None) -> tuple[int, int]:
    environ = os.environ if environ is None else environ
    return int(environ.get('JOB_COMPLETION_INDEX', '0')), int(environ.get('NUM_INSTANCES', '1'))


def _client(s3):
    if s3 is not None:
        return s3
    import boto3
    return boto3.client('s3')


def _missing(error: Exception) -> bool:
    code = str(getattr(error, 'response', {}).get('Error', {}).get('Code', ''))
    return code in {'NoSuchKey', '404', 'NotFound'}


def put_record(uri: str, payload: dict, s3=None) -> None:
    bucket, key = split_uri(uri)
    _client(s3).put_object(Bucket=bucket, Key=key, Body=(json.dumps(payload, indent=2) + '\n').encode(), ServerSideEncryption='AES256')


def get_record(uri: str, s3=None) -> dict | None:
    """The JSON record at ``uri``, or None when it does not exist (yet)."""
    bucket, key = split_uri(uri)
    try:
        return json.loads(_client(s3).get_object(Bucket=bucket, Key=key)['Body'].read())
    except Exception as error:
        if _missing(error):
            return None
        raise


def shared_job_network(rank: int, interface: str, port, prefix: str, deadline: float, *, s3=None, addresses=None,
                       sleep: Callable = time.sleep, clock: Callable = time.time) -> tuple[str, str, str]:
    """(local_ip, leader_ip, ray_address). Head: publish its address; peer: wait for it and match the address family."""
    local_addresses = addresses if addresses is not None else interface_addresses(interface)
    uri = f'{prefix}/{LEADER}'
    if rank == 0:
        local, _ = select_addresses(local_addresses, local_addresses)
        put_record(uri, {'ip': local, 'port': int(port), 'hostname': socket.gethostname(), 'at': clock()}, s3)
        return local, local, endpoint(local, port)
    while clock() < deadline:
        leader = get_record(uri, s3)
        if leader is not None:
            if int(leader['port']) != int(port):
                raise RuntimeError(f'Head rendezvous port mismatch: {leader["port"]} != {port}')
            local, ip = select_addresses(local_addresses, [routable(leader['ip'])])
            return local, ip, endpoint(ip, port)
        sleep(3)
    raise TimeoutError('Head did not publish its routable interface address before the startup deadline')


def start_ray(rank: int, address: str, port, local_ip: str, cpus: int, gpus: int, deadline: float, *, run: Callable = subprocess.run,
              sleep: Callable = time.sleep, attempts: int = 30) -> int:
    """Head: ``ray start --head``. Peer: ``ray start --address`` with up to ``attempts`` tries (the head may still be starting).
    Returns the number of tries used."""
    run([RAY, 'stop', '--force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    args = ray_start_args(rank, address, port, local_ip, cpus, gpus)
    if rank == 0:
        run(args, check=True, timeout=min(120, max(1, deadline - time.time())))
        return 1
    for trial in range(attempts):
        if time.time() >= deadline:
            raise TimeoutError('Ray startup exceeded the attempt deadline')
        try:
            result = run(args, timeout=min(45, max(1, deadline - time.time())))
        except subprocess.TimeoutExpired:
            sleep(2)
            continue
        if result.returncode == 0:
            return trial + 1
        sleep(min(3 + trial, 15))
    raise RuntimeError(f'Peer could not join the head Ray cluster at {address} after {attempts} tries')


def cluster_state(ray_module=None) -> dict:
    ray = ray_module or __import__('ray')
    alive = [n for n in ray.nodes() if n.get('Alive')]
    return {'alive': len(alive), 'gpus': int(ray.cluster_resources().get('GPU', 0)),
            'addresses': sorted({n.get('NodeManagerAddress') for n in alive})}


def wait_for_cluster(nodes: int, gpus_per_node: int, timeout: float, *, ray_module=None, sleep: Callable = time.sleep,
                     clock: Callable = time.time) -> dict:
    """Block until exactly ``nodes`` live Ray nodes on distinct addresses expose ``nodes * gpus_per_node`` GPUs."""
    until = clock() + timeout
    state = {}
    while clock() < until:
        state = cluster_state(ray_module)
        if state['alive'] == nodes and len(state['addresses']) == nodes and state['gpus'] == nodes * gpus_per_node:
            return state
        sleep(3)
    raise RuntimeError(f'Ray cluster did not reach {nodes} node(s) x {gpus_per_node} GPU(s) within {timeout:.0f}s: {state}')


def cluster_check(nodes: int, ray_module=None) -> Callable[[], str | None]:
    """For the head's Watcher: a message when fewer than ``nodes`` Ray nodes are alive."""
    def check():
        alive = cluster_state(ray_module)['alive']
        return None if alive >= nodes else f'only {alive} of {nodes} Ray nodes alive (a peer node was lost)'
    return check


class Watcher(threading.Thread):
    """Calls ``check`` every ``interval`` seconds; the first message it returns (twice in a row: one missed poll is tolerated)
    becomes ``failure`` and the training loop aborts on it."""

    def __init__(self, check: Callable[[], str | None], interval: float = 30.0, strikes: int = 2, error_strikes: int = 8):
        super().__init__(daemon=True)
        self.check, self.interval, self.strikes, self.error_strikes = check, interval, strikes, error_strikes
        self.failure: str | None = None
        self.errors = 0
        self._stop_event = threading.Event()

    def poll(self, misses: int) -> int:
        """A node reported lost twice in a row aborts; the check itself failing (a flaky Ray control plane) is NOT evidence of a lost node and
        is tolerated for ``error_strikes`` consecutive polls (4 min at 30 s) before it aborts."""
        try:
            message = self.check()
        except Exception as error:
            self.errors += 1
            if self.errors >= self.error_strikes:
                self.failure = f'liveness check failed {self.errors} times in a row: {type(error).__name__}: {error}'
            return misses
        self.errors = 0
        misses = misses + 1 if message else 0
        if message and misses >= self.strikes:
            self.failure = message
        return misses

    def run(self):
        misses = 0
        while not self._stop_event.wait(self.interval) and self.failure is None:
            misses = self.poll(misses)

    def stop(self):
        self._stop_event.set()

    def __call__(self) -> str | None:
        return self.failure


class Heartbeat(threading.Thread):
    """Writes ``{'at': now, **extra}`` to ``uri`` immediately and then every ``interval`` seconds until stopped."""

    def __init__(self, uri: str, extra: dict | None = None, interval: float = HEARTBEAT_SECONDS, s3=None):
        super().__init__(daemon=True)
        self.uri, self.extra, self.interval, self.s3 = uri, dict(extra or {}), interval, s3
        self._stop_event = threading.Event()

    def beat(self):
        try:
            put_record(self.uri, {'at': time.time(), **self.extra}, self.s3)
        except Exception as error:   # a transient S3 error must not kill the job; peers tolerate 300 s of silence
            print(f'heartbeat write failed: {type(error).__name__}: {error}', flush=True)

    def run(self):
        self.beat()
        while not self._stop_event.wait(self.interval):
            self.beat()

    def stop(self):
        self._stop_event.set()


def head_age(prefix: str, now: float, s3=None) -> float | None:
    """Seconds since the newest head record, or None when the head has written nothing yet."""
    times = [float(r['at']) for r in (get_record(f'{prefix}/{name}', s3) for name in HEAD_RECORDS) if r and 'at' in r]
    return now - max(times) if times else None


def peer_wait(prefix: str, rank: int, deadline: float, *, s3=None, identity: dict | None = None, poll: float = 10.0,
              heartbeat: float = HEARTBEAT_SECONDS, stale: float = HEAD_STALE_SECONDS, sleep: Callable = time.sleep,
              clock: Callable = time.time) -> dict:
    """Keep this peer's Ray node up until the head finishes. Returns the head's completion record; raises HeadStale when the
    head stopped writing records, TimeoutError at the attempt deadline."""
    beat_uri, last_beat, started = f'{prefix}/peer-{rank}-heartbeat.json', None, clock()
    last_ok = started
    while clock() < deadline:
        now = clock()
        try:   # one S3 hiccup on a peer must not take the whole job down: errors are tolerated for the stale window
            if last_beat is None or now - last_beat >= heartbeat:
                put_record(beat_uri, {'at': now, 'rank': rank, 'hostname': socket.gethostname(), **(identity or {})}, s3)
                last_beat = now
            final = get_record(f'{prefix}/completion.json', s3)
            if final is not None:
                return final
            age = head_age(prefix, now, s3)
            last_ok = now
        except Exception as error:
            print(f'peer {rank}: S3 record error (tolerated for {stale:.0f}s): {type(error).__name__}: {error}', flush=True)
            if now - last_ok > stale:
                raise HeadStale(f'S3 records unreadable for {now - last_ok:.0f}s (> {stale:.0f}s): leaving the cluster') from error
            sleep(poll)
            continue
        if (age if age is not None else now - started) > stale:
            raise HeadStale(f'head silent for {age if age is not None else now - started:.0f}s (> {stale:.0f}s): leaving the cluster')
        sleep(poll)
    raise TimeoutError('Peer reached the attempt deadline before the head wrote completion.json')
