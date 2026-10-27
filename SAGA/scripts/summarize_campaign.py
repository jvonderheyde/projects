"""Export every evaluated genome without loading MACE or its structures."""
import argparse,csv,json,math
from pathlib import Path

def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def summarize(root,out):
    out.mkdir(parents=True,exist_ok=True);runs=[];genomes=[];history=[]
    for path in sorted(root.glob('seed_*/checkpoint.json')):
        state=json.loads(path.read_text());cfg=state['config'];seed=cfg['seed'];report=path.with_name('report.json')
        runs.append(dict(seed=seed,generation=state['gen'],target_generations=cfg['n_generations'],unique_genomes=state['n_unique'],candidate_evaluations=state['n_evaluations'],validation_complete=state.get('validation_complete',False),report_present=report.exists(),elapsed_seconds=state.get('elapsed',0)))
        history.extend(dict(seed=seed,**{k:v for k,v in row.items() if k!='coverage'}) for row in state['history'])
        for phase in ('dataset','validation'):
            for r in state.get(phase,[]):
                elements={e for layer in r['counts'].values() for e,n in layer.items() if n}
                assessment=r['assessment'];row=dict(seed=seed,phase='search' if phase=='dataset' else 'validation',id=r['id'],facet=r['facet'],actual_k=len(elements),counts=json.dumps(r['counts'],sort_keys=True),n_evals=r['n_evals'],composite=assessment['composite'],eligible=assessment['eligible'],conservative_score=assessment['conservative_score'],all_mean_windows_met=assessment['all_mean_windows_met'])
                for ads in cfg['adsorbates']:
                    st=r['stats'][ads];lo,hi=cfg['objectives'][ads];mean=st['mean']
                    row.update({ads+'_mean_eV':mean,ads+'_se_eV':st['se'],ads+'_valid_fraction':st['valid_fraction'],ads+'_valid_configs':st['n_configs'],ads+'_attempted_probes':st['attempted'],ads+'_inside_window':mean is not None and lo<=mean<=hi,ads+'_rejections':json.dumps(st['rejections'],sort_keys=True)})
                genomes.append(row)
    write(out/'run_status.csv',runs);write(out/'genome_summary.csv',genomes);write(out/'generation_history.csv',history)
    print(json.dumps(runs,indent=2));return runs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('big_nh3_runs'));p.add_argument('--out',type=Path,default=Path('campaign_summary'));a=p.parse_args();summarize(a.root,a.out)
