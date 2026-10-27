"""Extract compact, path-free numerical examples from the earlier gp_results.zip."""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def extract(archive, destination):
    runs = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('/run/results.json') and '/__MACOSX/' not in n):
            data = json.loads(z.read(name))
            prefix = str(PurePosixPath(name).parent) + '/'
            timing = json.loads(z.read(prefix+'timing.json'))
            meta = data['manifest']
            config = {k: v for k, v in meta['config'].items() if k not in ('model', 'out', 'warm_start')}
            fields = ('genome', 'mean', 'se', 'valid', 'score', 'phase', 'generation',
                      'evaluation', 'facet', 'all_windows', 'worst_window_violation',
                      'n_valid_configs', 'valid_probe_fraction', 'evaluation_seconds')
            compact = lambda rows: [{k: r[k] for k in fields if k in r} for r in rows]
            run = dict(id=f"gp_k{config['k']}_seed{config['seed']}", config=config,
                       model_sha256=meta['model_sha256'], versions=meta['versions'],
                       adsorbates=meta['adsorbates'], windows_eV=meta['windows_eV'],
                       timing={k: timing[k] for k in ('complete', 'algorithm_elapsed_seconds',
                               'evaluated_candidates', 'proposals_scored', 'fit_predict_seconds', 'gpu_name')},
                       search=compact(data['evaluations']), validation=compact(data['validation']))
            runs.append(run)
    if not runs:
        raise ValueError('No supported historical results found')
    runs.sort(key=lambda r: (r['config']['k'], r['config']['seed']))
    summary = []
    for r in runs:
        valid = [v for v in r['validation'] if v['valid']]
        summary.append(dict(run=r['id'], k=r['config']['k'], seed=r['config']['seed'],
            complete=r['timing']['complete'], search_evaluations=len(r['search']),
            unique_search_genomes=len({tuple(v['genome']) for v in r['search']}),
            validation_finalists=len(r['validation']), eligible_finalists=len(valid),
            all_four_mean_hits=sum(bool(v.get('all_windows')) for v in valid),
            best_validation_fitness=max((v['score'] for v in valid), default=None),
            application_hours=r['timing']['algorithm_elapsed_seconds']/3600,
            surrogate_proposals_scored=r['timing']['proposals_scored']))
    destination.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    payload = dict(schema='historical-nh3-example-v1', source_archive=archive.name,
                   source_sha256=h.hexdigest(), protocol='Earlier toy GP implementation; not the release protocol',
                   runs=runs, summary=summary)
    (destination/'historical_nh3.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (destination/'run_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--out', type=Path, default=Path('examples/historical_nh3'))
    a = p.parse_args()
    extract(a.archive, a.out)
