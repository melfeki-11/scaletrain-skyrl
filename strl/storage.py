import json
from pathlib import Path
import boto3


def split_uri(uri):
    if not uri.startswith('s3://'):
        raise ValueError('Expected S3 URI')
    bucket, _, key = uri[5:].partition('/')
    return bucket, key.rstrip('/')


def upload_file(path, uri):
    bucket, key = split_uri(uri)
    boto3.client('s3').upload_file(str(path), bucket, key, ExtraArgs={'ServerSideEncryption': 'AES256'})


def put_json(uri, payload):
    bucket, key = split_uri(uri)
    boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=(json.dumps(payload, indent=2)+'\n').encode(), ServerSideEncryption='AES256')


def get_json(uri):
    bucket, key = split_uri(uri)
    return json.loads(boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read())


def sync_local(root, prefix, exclude=()):
    """Upload new/changed files under ``root`` to ``prefix``. Skips every path with a component starting with '.' (SkyRL stages HF exports
    in ``.tmp-hf-export-*`` and renames them into place: uploading the staging dir raced the rename and raised FileNotFoundError), and files
    that vanish between listing and upload. Returns the number of files uploaded."""
    root = Path(root)
    if not root.exists():
        return 0
    files = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if any(part.startswith('.') for part in rel.parts):
            continue
        if exclude and any(part in str(rel) for part in exclude):
            continue
        try:
            if path.is_file():
                stat = path.stat()
                files.append((path, (stat.st_size, stat.st_mtime_ns)))
        except FileNotFoundError:
            continue
    cache = getattr(sync_local, '_cache', {})
    uploaded = 0
    for path, signature in files:
        target = prefix.rstrip('/')+'/'+str(path.relative_to(root))
        if cache.get(target) != signature:
            try:
                upload_file(path, target)
            except FileNotFoundError:
                continue
            cache[target] = signature
            uploaded += 1
    sync_local._cache = cache
    return uploaded


def sync_cluster(root, prefix, timeout=1800):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    nodes = [n for n in ray.nodes() if n['Alive']]
    @ray.remote(num_cpus=0)
    def sync_one(root, prefix):
        sync_local(root, prefix)
        return {'node_id': ray.get_runtime_context().get_node_id(), 'synced': True}
    return ray.get([sync_one.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n['NodeID'],soft=False)).remote(root,prefix) for n in nodes], timeout=timeout)


def download_prefix(prefix, root):
    bucket, key = split_uri(prefix)
    s3 = boto3.client('s3')
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=key+'/'):
        for item in page.get('Contents', []):
            relative = item['Key'][len(key)+1:]
            path = Path(root)/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.is_file() or path.stat().st_size != item['Size']:
                s3.download_file(bucket,item['Key'],str(path))
