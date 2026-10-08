#!/usr/bin/env python3
"""Read-only EKS capacity routing for configured teams and H100 presets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path
import re

TEAMS=['gen_ai.agent_training_foundation','gen_ai.agents_research']
RESOURCES=['nvidia.com/gpu','cpu','memory','ephemeral-storage','vpc.amazonaws.com/efa']


def quantity(value):
    match=re.fullmatch(r'([0-9.+-]+)([A-Za-z]*)',str(value))
    if not match:raise ValueError('Unsupported Kubernetes quantity')
    value,suffix=match.groups()
    factors={'':1,'m':Decimal('.001'),'Ki':1024,'Mi':1024**2,'Gi':1024**3,'Ti':1024**4,'Pi':1024**5,'K':1000,'M':1000**2,'G':1000**3,'T':1000**4}
    return Decimal(value)*factors[suffix]


def pod_requests(pod):
    spec=pod['spec'];app=defaultdict(Decimal);init_peak=defaultdict(Decimal);sidecars=defaultdict(Decimal)
    for container in spec.get('containers',[]):
        for name,value in container.get('resources',{}).get('requests',{}).items():app[name]+=quantity(value)
    for container in spec.get('initContainers',[]):
        request={k:quantity(v) for k,v in container.get('resources',{}).get('requests',{}).items()}
        if container.get('restartPolicy')=='Always':
            for name,value in request.items():sidecars[name]+=value
        for name in set(request)|set(sidecars):init_peak[name]=max(init_peak[name],sidecars[name]+(0 if container.get('restartPolicy')=='Always' else request.get(name,0)))
    return {name:max(app[name]+sidecars[name],init_peak[name])+quantity(spec.get('overhead',{}).get(name,0)) for name in set(app)|set(init_peak)|set(sidecars)|set(spec.get('overhead',{}))}


def flavor_quota(queue,flavor):
    return {r['name']:r for group in queue['spec'].get('resourceGroups',[]) for f in group['flavors'] if f['name']==flavor for r in f['resources']}


def reservations(queue,flavor):
    return {r['name']:quantity(r['total']) for f in queue.get('status',{}).get('flavorsReservation',[]) if f['name']==flavor for r in f['resources']}


def active(queue):
    return queue['spec'].get('stopPolicy','None')=='None' and any(c['type']=='Active' and c['status']=='True' for c in queue.get('status',{}).get('conditions',[]))


def node_slots(snapshot,gpus,node_selector=None,cpu_per_node=None):
    # These shapes were observed from actual scheduler pod resource requests.
    request={'nvidia.com/gpu':Decimal(gpus),'cpu':Decimal(18*gpus),'memory':quantity(f'{192*gpus}Gi'),'ephemeral-storage':quantity(f'{384*gpus if gpus<8 else 3073}Gi')}
    if cpu_per_node is not None:request['cpu']=quantity(cpu_per_node)
    if gpus==8:request['vpc.amazonaws.com/efa']=Decimal(32)
    used=defaultdict(lambda:defaultdict(Decimal))
    for pod in snapshot['pods']:
        if pod['status']['phase'] in {'Succeeded','Failed'}:continue
        for name,value in pod_requests(pod).items():used[pod['spec'].get('nodeName')][name]+=value
    groups=defaultdict(list)
    for node in snapshot['nodes']:
        labels=node['metadata'].get('labels',{});spec=node.get('spec',{})
        if any(labels.get(k)!=v for k,v in (node_selector or {}).items()):continue
        if labels.get('node.kubernetes.io/instance-type')!='p5.48xlarge' or spec.get('unschedulable'):continue
        if not any(c['type']=='Ready' and c['status']=='True' for c in node['status'].get('conditions',[])):continue
        if any(t['effect'] in {'NoSchedule','NoExecute'} and t['key']!='nvidia.com/gpu' for t in spec.get('taints',[]) or []):continue
        name=node['metadata']['name'];alloc=node['status']['allocatable']
        if not all(quantity(alloc.get(resource,0))-used[name][resource]>=value for resource,value in request.items()):continue
        domain=labels.get('topology.k8s.aws/network-node-layer-2')
        if domain:groups[domain].append(name)
    return request,dict(groups)


def plan(snapshot,nodes=2,shapes=(8,4,2,1),distinct_placement_verified=False,node_selector=None,cpu_per_node=None):
    queues={q['metadata']['name']:q for q in snapshot['clusterqueues']}
    routes=[];selected=None
    # Non-borrowing globally first; team preference applies within each mode.
    for borrow in [False,True]:
        for team in TEAMS:
            name=team.replace('_','-')+('-borrowing' if borrow else '')
            queue=queues.get(name)
            for gpus in shapes:
                selector={'k8s.amazonaws.com/accelerator':'nvidia-hopper-h100','manually_preemptible':'false'}
                for key,value in (node_selector or {}).items():
                    if value is None or value == "":selector.pop(key,None)
                    else:selector[key]=value
                request,groups=node_slots(snapshot,gpus,selector,cpu_per_node);resources={k:v*nodes for k,v in request.items()}
                reasons=[];headroom={}
                if gpus<8 and nodes>1 and not distinct_placement_verified:reasons.append('sliced preset co-locates replicas unless distinct-host placement is explicitly verified')
                if queue is None or not active(queue):reasons.append('queue unavailable or inactive')
                else:
                    quotas=flavor_quota(queue,'h100');reserved=reservations(queue,'h100')
                    for resource,need in resources.items():
                        own=quantity(quotas.get(resource,{}).get('nominalQuota',0))-reserved.get(resource,0)
                        available=own
                        if borrow:
                            cohort=queue['spec'].get('cohort')
                            peers=[q for q in queues.values() if cohort and q['spec'].get('cohort')==cohort]
                            shared=sum((quantity(flavor_quota(q,'h100').get(resource,{}).get('nominalQuota',0))-reservations(q,'h100').get(resource,0) for q in peers),Decimal(0))
                            limit=quotas.get(resource,{}).get('borrowingLimit')
                            available=own+max(Decimal(0),shared)
                            if limit is not None:available=min(available,own+quantity(limit))
                        headroom[resource]=str(max(0,available))
                        if available<need:reasons.append(resource+' quota insufficient')
                domains={d:n for d,n in groups.items() if len(n)>=nodes}
                if not domains:reasons.append('not enough distinct ready nodes in one network domain')
                route={'team':team,'allow_borrowing':borrow,'backend':'eks','cluster':snapshot.get('cluster','ml-gpu-batch'),'instance_type':'p5.48xlarge'+(f':{gpus}' if gpus<8 else ''),'gpus_per_node':gpus,'nodes':nodes,'queue':name,'headroom':headroom,'requested_per_node':{k:str(v) for k,v in request.items()},'node_selector':selector,'eligible':not reasons,'reasons':reasons,'compatible_nodes':domains}
                routes.append(route)
                if selected is None and not reasons:selected=route
    return {'checked_at':snapshot.get('checked_at'),'selected':selected,'routes':routes,'capacity_is_estimate':True}


def plan_many(snapshots,verified_launch_clusters=('ml-gpu-batch',),**kwargs):
    """Compare clusters without letting cluster order override team/borrowing order."""
    routes=[];checks=[];verified=set(verified_launch_clusters)
    for snapshot in snapshots:
        cluster=snapshot.get('cluster')
        if not cluster:raise ValueError('Multi-cluster snapshots require an explicit cluster identity')
        if snapshot.get('error_type'):
            checks.append({'cluster':cluster,'error_type':snapshot['error_type'],'launch_route_verified':cluster in verified})
            continue
        result=plan(snapshot,**kwargs)
        for route in result['routes']:
            route['capacity_eligible']=route['eligible']
            route['launch_route_verified']=cluster in verified
            if cluster not in verified:
                route['eligible']=False
                route['reasons'].append('launch routing to this cluster is not verified')
        routes.extend(result['routes'])
        gpu_types=sorted({n['metadata'].get('labels',{}).get('node.kubernetes.io/instance-type','unknown') for n in snapshot['nodes'] if quantity(n['status'].get('allocatable',{}).get('nvidia.com/gpu',0))>0})
        checks.append({'cluster':cluster,'checked_at':snapshot.get('checked_at'),'launch_route_verified':cluster in verified,'gpu_types':gpu_types})
    shapes=tuple(kwargs.get('shapes',(8,4,2,1)))
    cluster_order={s['cluster']:i for i,s in enumerate(snapshots)}
    routes.sort(key=lambda r:(r['allow_borrowing'],TEAMS.index(r['team']),shapes.index(r['gpus_per_node']),cluster_order[r['cluster']]))
    return {'selected':next((r for r in routes if r['eligible']),None),'routes':routes,'cluster_checks':checks,'capacity_is_estimate':True}


def collect(profile,region,cluster):
    import base64,ssl,subprocess,tempfile,urllib.request
    import boto3
    session=boto3.Session(profile_name=profile)
    info=session.client('eks',region_name=region).describe_cluster(name=cluster)['cluster']
    token=json.loads(subprocess.check_output(['aws','--profile',profile,'eks','get-token','--cluster-name',cluster,'--region',region],text=True))['status']['token']
    with tempfile.NamedTemporaryFile() as ca:
        ca.write(base64.b64decode(info['certificateAuthority']['data']));ca.flush();context=ssl.create_default_context(cafile=ca.name)
        def get(path):
            request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+token})
            with urllib.request.urlopen(request,context=context,timeout=30) as response:return json.load(response)['items']
        queues=get('/apis/kueue.x-k8s.io/v1beta1/clusterqueues');nodes=get('/api/v1/nodes');pods=get('/api/v1/pods')
    # Do not retain commands, environment variables, volumes or unrelated payloads.
    pods=[{'spec':{'nodeName':p['spec'].get('nodeName'),'containers':[{'resources':c.get('resources',{})} for c in p['spec']['containers']],'initContainers':[{'resources':c.get('resources',{}),'restartPolicy':c.get('restartPolicy')} for c in p['spec'].get('initContainers',[])],'overhead':p['spec'].get('overhead',{})},'status':{'phase':p['status']['phase']}} for p in pods]
    return {'checked_at':datetime.now(timezone.utc).isoformat(),'cluster':cluster,'profile':profile,'region':region,'clusterqueues':queues,'nodes':nodes,'pods':pods}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--distinct-placement-verified',action='store_true');parser.add_argument('--cpu-per-node');parser.add_argument('--node-selector',action='append',default=[]);parser.add_argument('--snapshot');parser.add_argument('--output',required=True);parser.add_argument('--nodes',type=int,default=2);parser.add_argument('--gpus-per-node',default='8,4,2,1');parser.add_argument('--profile',default='ml-admin');parser.add_argument('--region',default='us-west-2');parser.add_argument('--cluster',default='ml-gpu-batch');parser.add_argument('--clusters',help='Comma-separated EKS clusters to inspect concurrently');parser.add_argument('--verified-launch-clusters',default='ml-gpu-batch',help='Clusters with independently verified launcher, image, assets and team routing');args=parser.parse_args()
    if args.nodes<1:parser.error('--nodes must be positive')
    shapes=tuple(int(s) for s in args.gpus_per_node.split(','))
    if not shapes or len(set(shapes))!=len(shapes) or any(s not in {1,2,4,8} for s in shapes):parser.error('Supported GPU counts per node are 1,2,4,8 without duplicates')
    clusters=list(dict.fromkeys((args.clusters or args.cluster).split(',')))
    if args.snapshot:
        loaded=json.loads(Path(args.snapshot).read_text());snapshots=loaded if isinstance(loaded,list) else [loaded]
        for snapshot in snapshots:
            if len(snapshots)==1:snapshot.setdefault('cluster',args.cluster)
    else:
        def inspect(cluster):
            try:return collect(args.profile,args.region,cluster)
            except Exception as error:return {'cluster':cluster,'error_type':type(error).__name__}
        with ThreadPoolExecutor(max_workers=min(4,len(clusters))) as pool:snapshots=list(pool.map(inspect,clusters))
    result=plan_many(snapshots,tuple(args.verified_launch_clusters.split(',')),nodes=args.nodes,shapes=shapes,distinct_placement_verified=args.distinct_placement_verified,node_selector=dict(s.split('=',1) for s in args.node_selector),cpu_per_node=args.cpu_per_node)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps({'selected':result['selected'],'output':str(Path(args.output).resolve())}))

if __name__=='__main__':main()
