"""Frozen, locked Slurm campaign launcher: new, resume, or skip complete."""
import argparse,dataclasses,fcntl,hashlib,json,os,signal,sys
from pathlib import Path
from .config import Config
from .evaluation_cache import atomic_json

PERFORMANCE={'max_batch_atoms','max_batch_edges','autotune','progress','device'}
VOLATILE=PERFORMANCE|{'resume','out_dir','model_path','n_generations'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def source_identity():return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def semantic_config(cfg):return json.loads(json.dumps({k:v for k,v in cfg.to_dict().items() if k not in VOLATILE}))

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--model',type=Path)
    p.add_argument('--root',type=Path,default=Path('big_nh3_runs'));p.add_argument('--recipe',type=Path,default=Path('campaign.json'))
    p.add_argument('--generations',type=int);p.add_argument('--max-batch-atoms',type=int);p.add_argument('--max-batch-edges',type=int)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke-test',action='store_true')
    return p.parse_args(argv)

def launch(args):
    recipe=json.loads(args.recipe.read_text());reaction=recipe.pop('reaction');recipe.update(seed=args.seed)
    if reaction not in {'nh3','nh3_full'}:
        raise ValueError('the release launcher supports nh3 and nh3_full recipes')
    for key in ('generations','max_batch_atoms','max_batch_edges'):
        value=getattr(args,key)
        if value is not None:recipe['n_generations' if key=='generations' else key]=value
    out=args.root/f'seed_{args.seed}';recipe.update(out_dir=str(out))
    if args.smoke_test:
        recipe.update(demo=True,device='cpu',bootstrap_size=8,batch_size=3,n_generations=args.generations if args.generations is not None else 2,offspring_size=24,population_size=8,elite_size=2,n_configs=2,n_probes=2,finalist_count=2,finalist_repeats=1,validation_n_probes=3,collect=False)
        model={'kind':'synthetic-demo-v1'}
    else:
        if args.model is None or not args.model.is_file():raise ValueError('--model must point to the trained MACE model')
        recipe['model_path']=str(args.model.resolve());model={'sha256':digest(args.model),'size_bytes':args.model.stat().st_size}
    cfg=Config.for_reaction(reaction,**recipe)
    identity=dict(schema='big-nh3-campaign-v1',source=source_identity(),model=model,config=semantic_config(cfg))
    manifest=out/'campaign_manifest.json';checkpoint=out/'checkpoint.json';report=out/'report.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old!=json.loads(json.dumps(identity)):raise ValueError('frozen campaign model/source/scientific settings differ; use a fresh root')
    elif out.exists() and any(out.iterdir()):raise ValueError('output is not an empty or recognized campaign directory')
    state=None
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        if state.get('version')!=3 or state.get('source_identity')!=identity['source'] or state.get('model_identity')!=model:
            raise ValueError('checkpoint identity or schema differs from campaign')
        if semantic_config(Config(**state['config']))!=semantic_config(cfg):raise ValueError('checkpoint scientific settings differ')
        if state['gen']>cfg.n_generations:raise ValueError('cannot reduce target below checkpoint generation')
    done=False
    if report.exists():
        r=json.loads(report.read_text())
        if state is None or r['run_id']!=state['run_id'] or r['source_identity']!=identity['source'] or r['model_identity']!=model:
            raise ValueError('report identity differs from checkpoint')
        done=r['final_generation']>=cfg.n_generations and bool(state.get('validation_complete'))
    action='SKIP complete' if done else ('RESUME' if checkpoint.exists() else 'START')
    print(f'{action}: seed={cfg.seed}, target_gen={cfg.n_generations}, out={out}',flush=True)
    if args.dry_run:
        print(json.dumps(cfg.to_dict(),indent=2));return None
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('another job already owns this seed directory')
        # Recheck after locking: simultaneous submissions may have been waiting on startup.
        if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(identity)):raise ValueError('campaign identity changed while acquiring lock')
        if not manifest.exists():atomic_json(manifest,identity)
        if done:return json.loads(report.read_text())
        if report.exists() and checkpoint.exists():
            fresh=json.loads(report.read_text());latest=json.loads(checkpoint.read_text())
            if fresh.get('run_id')!=latest.get('run_id') or latest.get('source_identity')!=identity['source'] or latest.get('model_identity')!=model:
                raise ValueError('run identity changed while acquiring lock')
            if fresh['final_generation']>=cfg.n_generations and latest.get('validation_complete'):
                print('SKIP complete after acquiring lock',flush=True);return fresh
        cfg.resume=checkpoint.exists()
        from .optimizer import ActiveLoop
        loop=ActiveLoop(cfg)
        result=loop.run()
        atomic_json(out/'effective_config.json',cfg.to_dict())
        return result

def main(argv=None):
    def stop(signum,frame):raise KeyboardInterrupt('Slurm termination; committed candidate caches will be reused')
    signal.signal(signal.SIGTERM,stop)
    try:launch(parse_args(argv))
    except KeyboardInterrupt as e:
        print(str(e),file=sys.stderr);return 143
    except (OSError,ValueError,RuntimeError,TypeError) as e:
        print('ERROR:',str(e),file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
