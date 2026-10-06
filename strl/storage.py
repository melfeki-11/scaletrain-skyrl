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
    root = Path(root)
    files = []
    for path in root.rglob('*'):
        if exclude and any(part in str(path.relative_to(root)) for part in exclude):
            continue
        if path.is_file() and not path.name.startswith('.'):
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            files.append((path, signature))
    cache = getattr(sync_local, '_cache', {})
    for path, signature in files:
        target = prefix.rstrip('/')+'/'+str(path.relative_to(root))
        if cache.get(target) != signature:
            upload_file(path, target)
            cache[target] = signature
    sync_local._cache = cache


def sync_cluster(root, prefix):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    nodes = [n for n in ray.nodes() if n['Alive']]
    @ray.remote(num_cpus=0)
    def sync_one(root, prefix):
        sync_local(root, prefix)
        return {'node_id': ray.get_runtime_context().get_node_id(), 'synced': True}
    return ray.get([sync_one.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n['NodeID'],soft=False)).remote(root,prefix) for n in nodes], timeout=240)


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
