"""Scale LiteLLM proxy client (stdlib HTTP, OpenAI wire format). Used by the qualitative critic; never by rewards.

Key resolution (never printed, never put in argv/config/tables):
  1. $LITELLM_PROXY_API_KEY
  2. $STRL_LITELLM_KEY_FILE (mode-600 file written by the worker)
  3. AWS Secrets Manager ``team/GENAIML/secret-store-key`` field ``HIL_BENCH_EVAL_1`` with ambient credentials
     (ScaleTrain worker role), else by assuming ``arn:aws:iam::307185671274:role/ml-worker`` (the ScaleTrain/litellm_test.py path)
Every request is attributed to a 24-hex Scale project id (user + metadata.tags).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = 'https://litellm-proxy.ml-serving-internal.scale.com'
SECRET_ID = 'team/GENAIML/secret-store-key'
KEY_NAME = 'HIL_BENCH_EVAL_1'
ML_WORKER_ROLE = 'arn:aws:iam::307185671274:role/ml-worker'
PROJECT_ID = '696011502a26dc4a64b24f55'
PROJECT_RE = re.compile(r'^[0-9a-f]{24}$')


class LiteLLMError(RuntimeError):
    pass


def _secret_store_value(name: str, region: str = 'us-west-2') -> str | None:
    import boto3
    sessions = [boto3.Session()]
    try:
        sts = boto3.client('sts')
        if sts.get_caller_identity()['Account'] != ML_WORKER_ROLE.split(':')[4]:
            creds = sts.assume_role(RoleArn=ML_WORKER_ROLE, RoleSessionName='strl-critic')['Credentials']
            sessions.append(boto3.Session(aws_access_key_id=creds['AccessKeyId'], aws_secret_access_key=creds['SecretAccessKey'],
                                          aws_session_token=creds['SessionToken']))
    except Exception:
        pass
    for session in sessions:
        try:
            raw = session.client('secretsmanager', region_name=region).get_secret_value(SecretId=SECRET_ID)['SecretString']
            try:
                store = json.loads(raw)
            except json.JSONDecodeError:
                import yaml
                store = yaml.safe_load(raw)
            if isinstance(store, dict) and store.get(name):
                return str(store[name])
        except Exception:
            continue
    return None


def resolve_key(name: str = KEY_NAME) -> str:
    key = os.environ.get('LITELLM_PROXY_API_KEY')
    if key:
        return key
    path = os.environ.get('STRL_LITELLM_KEY_FILE')
    if path and Path(path).is_file():
        return Path(path).read_text().strip()
    key = _secret_store_value(name)
    if key:
        return key
    raise LiteLLMError(f'LiteLLM key {name} unavailable (env, key file, secret store, ml-worker role all failed)')


def write_key_file(path: str | Path, name: str = KEY_NAME) -> Path:
    """Resolve the key once and write it mode-600 (so Ray actors read a file path, not a value in their env)."""
    key = resolve_key(name)
    target = Path(path)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(key)
    return target


def chat(messages: list[dict], *, model: str = 'openai/gpt-6-luna', max_completion_tokens: int = 1024, project_id: str = PROJECT_ID,
         base_url: str = BASE_URL, timeout: float = 60.0, json_mode: bool = True, key: str | None = None) -> dict:
    """One chat completion. Returns {'text', 'cost_usd' (proxy-reported when available), 'usage'}. No retries."""
    if not PROJECT_RE.fullmatch(project_id):
        raise LiteLLMError('project_id must be a 24-character hex ObjectId')
    body = {'model': model, 'messages': messages, 'max_completion_tokens': max_completion_tokens, 'user': project_id,
            'metadata': {'tags': [f'project:{project_id}', 'strl-critic']}}
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    request = urllib.request.Request(base_url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (key or resolve_key())})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            cost = response.headers.get('x-litellm-response-cost')
    except urllib.error.HTTPError as error:  # body may echo request details: report status only
        raise LiteLLMError(f'LiteLLM HTTP {error.code}') from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise LiteLLMError(f'LiteLLM unreachable: {type(error).__name__}') from None
    choice = (payload.get('choices') or [{}])[0]
    return {'text': (choice.get('message') or {}).get('content') or '', 'usage': payload.get('usage') or {},
            'cost_usd': float(cost) if cost not in (None, '') else None}


def parse_json_object(text: str) -> dict | None:
    """First JSON object in ``text`` (models sometimes wrap JSON in prose or fences)."""
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
