"""Execute one candidate program against its unit tests, defensively. Runs inside the Modal executor container (and locally in
unit tests). Pure stdlib.

Untrusted-code rules (the model is optimised AGAINST this verifier, so every shortcut gets found eventually):
  * the candidate never sees the tests or the pass token: both arrive on stdin and are read by the harness BEFORE the candidate
    executes; the harness reports through a private duplicate of fd 1 captured before the candidate runs;
  * a pass requires the harness to finish and print a fresh per-call nonce after the tests: ``sys.exit(0)`` / ``os._exit(0)`` /
    crashing the interpreter early can never count as passing;
  * each program runs in its own temp dir and session, as ``nobody`` when root, with CPU-time, memory, file-size and process
    limits plus a wall-clock limit; the container itself has no network and no Modal API access.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time

HARNESS = r'''
import ast, contextlib, io, json, os, sys, types
_out = os.fdopen(os.dup(1), 'w')


def _strl_eq(a, b):
    # equality asserts may not be satisfied by a class whose __eq__ the CANDIDATE wrote (the classic RL reward hack);
    # stdlib types implemented in Python (collections.Counter, fractions.Fraction, ...) stay legitimate
    for x in (a, b):
        for klass in type(x).__mro__:
            eq = klass.__dict__.get('__eq__')
            if isinstance(eq, types.FunctionType) and eq.__code__.co_filename == '<solution>':
                raise AssertionError('candidate-defined __eq__ rejected')
    return a == b


class _Strict(ast.NodeTransformer):
    def visit_Assert(self, node):
        self.generic_visit(node)
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            node.test = ast.Call(func=ast.Name('_strl_eq', ast.Load()), args=[t.left, t.comparators[0]], keywords=[])
        return node


def _compile_test(source, name):
    tree = _Strict().visit(ast.parse(source))
    return compile(ast.fix_missing_locations(tree), name, 'exec')


_dumps = json.dumps
job = json.loads(sys.stdin.read())
nonce = job.pop('nonce')
res = {'compiled': False, 'passed': 0, 'total': len(job['tests']), 'error': None, 'detail': None, 'first_failure': None}
ns = {'__name__': '__main__', '__builtins__': __builtins__}
sink = io.StringIO()
try:
    code = compile(job['setup'] + '\n' + job['code'], '<solution>', 'exec')
    res['compiled'] = True
except (SyntaxError, ValueError) as e:
    res['error'] = 'syntax_error'
    res['detail'] = (type(e).__name__ + ': ' + str(e))[:300]
if res['compiled']:
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(code, ns)
    except BaseException as e:
        res['error'] = 'runtime_error'
        res['detail'] = ('import/definition: ' + type(e).__name__ + ': ' + str(e))[:300]
    else:
        for i, test in enumerate(job['tests']):
            try:
                ns['_strl_eq'] = _strl_eq
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    exec(_compile_test(test, '<test%d>' % i), ns)
                res['passed'] += 1
            except AssertionError:
                if res['first_failure'] is None:
                    res['first_failure'] = ('test %d failed: ' % i) + test[:200]
            except BaseException as e:
                if res['error'] is None:
                    res['error'] = 'runtime_error'
                    res['detail'] = ('test %d: ' % i + type(e).__name__ + ': ' + str(e))[:300]
_out.write(_dumps(res) + '\n' + nonce + '\n')
_out.flush()
os._exit(0)
'''

VERSION = hashlib.sha256((HARNESS + 'limits-v2').encode()).hexdigest()[:16]  # deployed executor must match
SUSPICIOUS = re.compile(r'__eq__|__ne__|sys\.exit|os\._exit|\bexit\(|\bquit\(|atexit|import\s+builtins|__builtins__|sys\.modules|'
                        r'inspect\.|_getframe|gc\.get_objects|signal\.')
FAILURE_CLASSES = ('no_code_block', 'syntax_error', 'runtime_error', 'wrong_answer', 'timeout', 'infra_error')


def _limits(cpu_seconds: int, memory_mb: int):
    def apply():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        memory = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (ValueError, OSError):  # macOS does not support RLIMIT_AS
            pass
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        if os.geteuid() == 0:
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
                os.setgid(65534)
                os.setuid(65534)
            except OSError:
                pass
    return apply


def run_program(job: dict, *, python: str | None = None, cpu_seconds: int = 6, wall_seconds: float = 12.0, memory_mb: int = 1024) -> dict:
    """job = {'code', 'tests': [statement...], 'setup': str}. Returns the verdict dict (never raises for candidate behaviour)."""
    started = time.monotonic()
    nonce = secrets.token_hex(16)
    payload = json.dumps({'nonce': nonce, 'code': job['code'], 'tests': list(job['tests']), 'setup': job.get('setup', '')})
    verdict = {'passed': 0, 'total': len(job['tests']), 'failure_class': None, 'detail': None, 'nonce_ok': False, 'compiled': False,
               'harness_version': VERSION}
    workdir = tempfile.mkdtemp(prefix='strl-exec-')
    os.chmod(workdir, 0o777)
    try:
        process = subprocess.Popen([python or sys.executable, '-I', '-c', HARNESS], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=workdir, env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0', 'HOME': workdir},
                                   preexec_fn=_limits(cpu_seconds, memory_mb))
        try:
            stdout, stderr = process.communicate(payload.encode(), timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            verdict.update(failure_class='timeout', detail=f'wall clock > {wall_seconds}s')
            return verdict
        lines = stdout.decode(errors='replace').strip().splitlines()
        if len(lines) >= 2 and lines[-1] == nonce:
            result = json.loads(lines[-2])
            verdict.update(nonce_ok=True, passed=result['passed'], compiled=result['compiled'], detail=result['detail'] or result['first_failure'])
            if result['error'] == 'syntax_error':
                verdict['failure_class'] = 'syntax_error'
            elif result['total'] > 0 and result['passed'] == result['total']:
                verdict['failure_class'] = 'ok'
            elif result['error'] == 'runtime_error' and result['passed'] == 0 and result['first_failure'] is None:
                verdict['failure_class'] = 'runtime_error'
            else:
                verdict['failure_class'] = 'wrong_answer'
        elif process.returncode in (-signal.SIGXCPU, -signal.SIGKILL) or process.returncode == 152:
            verdict.update(failure_class='timeout', detail=f'cpu limit {cpu_seconds}s')
        else:  # exited before the harness reported: sys.exit/os._exit/crash in the candidate. Never a pass.
            verdict.update(failure_class='runtime_error', detail=f'exited early rc={process.returncode}: {stderr.decode(errors="replace")[-300:]}')
        return verdict
    finally:
        verdict['latency_s'] = round(time.monotonic() - started, 4)
        subprocess.run(['rm', '-rf', workdir], check=False)


def suspicious(code: str | None) -> bool:
    return bool(code) and bool(SUSPICIOUS.search(code))
