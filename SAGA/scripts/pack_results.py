"""Pack analysis results without model files, geometry caches or collected structures."""
import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('big_nh3_runs'));p.add_argument('--out',type=Path,default=Path('big_nh3_results.zip'));a=p.parse_args()
keep={'checkpoint.json','report.json','run_log.jsonl','campaign_manifest.json','effective_config.json','gas_references.json'}
with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
    for seed in sorted(a.root.glob('seed_*')):
        for f in sorted(seed.iterdir()):
            if f.is_file() and f.name in keep:z.write(f,str(Path(a.root.name)/seed.name/f.name))
with zipfile.ZipFile(a.out) as z:
    if z.testzip() is not None:raise ValueError('results ZIP integrity check failed')
print(a.out.resolve())
