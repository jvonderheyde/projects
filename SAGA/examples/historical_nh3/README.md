# Historical four-window example

These compact values come from six completed earlier MACE GP searches: exact k=2,3,5, seeds 123/124, 36 generations, three configurations and three probes/species during search, eight independently sampled finalists per run. They are not outputs of the new release protocol. Original numerical means, standard errors, validity flags and recorded fitness are retained; structures and per-probe diagnostics are excluded.

`run_summary.csv` is ready for a spreadsheet. `historical_nh3.json` contains the reduced numerical records, original package versions, model hash and input-archive SHA-256. No machine-specific paths are included. The extraction script can recreate it from the original archive:

```bash
python scripts/extract_historical.py /path/to/gp_results.zip
```

Open `historical_results.ipynb` to customize colors, subsets and statistics. Its PNG compares per-generation **new-candidate** fitness and independently validated mean-window hits. Faded bands show sample standard deviation across two seeds; two seeds are insufficient to establish a reliable ranking of k.

All-four hits use the original validity flag and inclusive numerical mean windows. They do not require uncertainty intervals to lie inside all windows. Historical contact/reference policies differ from the new presets. The source measured application runtime across resumes; do not interpret first/last timestamps as active GPU time or report proposed surrogate genomes as MACE evaluations.
