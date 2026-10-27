# Full NH₃ SAGA campaign

Five independent GP searches, seeds 123–127, over globally capped ≤10-element, 27-atom pseudo-fcc slabs on (100)/(111). All six existing NH₃ windows and gas references are retained. This package includes the full SAGA core, the frozen campaign recipe, resume handling and result-export tools. It does not include a MACE model, NEB or a microkinetic model.

## Run

Use the existing CUDA/PyTorch/MACE environment, or install `requirements.txt` with an appropriate CUDA PyTorch installation. Put the trained model at `models/s8.model`, or provide its actual path below. Submit **from this extracted directory**:

```bash
sbatch scripts/submit_big_saga.sh 123 models/s8.model
sbatch scripts/submit_big_saga.sh 124 models/s8.model
sbatch scripts/submit_big_saga.sh 125 models/s8.model
sbatch scripts/submit_big_saga.sh 126 models/s8.model
sbatch scripts/submit_big_saga.sh 127 models/s8.model
```

These are five separate jobs, not an array or a nested submitter. Each requests one GPU, eight CPUs, 32 GB host RAM and 24 hours. Adjust the `#SBATCH` resource lines for the cluster. The walltime is a resource request, not a runtime prediction. Output defaults to `big_nh3_runs/seed_123`, etc.; logs stay in the submission directory. No working-directory or Slurm spool paths are inferred.

An optional third argument changes the output root:

```bash
sbatch scripts/submit_big_saga.sh 123 models/s8.model /scratch/project/big_nh3_runs
```

Edit `campaign.json` before launching fresh runs. Every scientific setting, the Python source and the model SHA-256 are frozen per seed. Paths alone do not identify a model. Check an explicit model path and effective recipe without loading MACE or starting a search:

```bash
DRY_RUN=1 bash scripts/submit_big_saga.sh 123 models/s8.model
```

## Search recipe

| Setting | Value |
|---|---|
| Reaction | `nh3_full`: H, N, N₂, NH, NH₂, NH₃ |
| Element pool | Ag, Au, Cu, Fe, Mo, Ni, Pd, Pt, Sn, Zn |
| Components | Global cap of 10; actual k can be 1–10 |
| Slab | 3×3×3; bottom layer fixed; cell fixed |
| Generations / bootstrap / new batch | 60 / 48 / 8 |
| Population / proposed offspring | 32 / 512 |
| Search sampling | 4 arrangements × 3 probes/species |
| Slots per new batch | 1 coverage + 1 window + 6 acquisition/diversity |
| Elite refinements | At most 1 per generation |
| Weights | N=2, NH₂=2; H=N₂=NH=NH₃=1 |
| Clean reference | Lowest valid converged result from 3 starts: nominal + 2 free-atom jitters, σ=0.03 Å |
| Contact screen | At least one original adsorbate/slab pair within 1.25 × summed ASE covalent radii, using periodic distances |
| Relaxation | LBFGS, literal 500 steps maximum; slab/ads fmax=0.05, gas fmax=0.03 eV/Å |
| Precision | CUDA float64 |
| Independent validation | Up to 10 finalists/seed; 5 fresh batches × 4 configurations × 5 probes/species |

Targets are 528 unique genomes/seed, 2,640 overall, before cross-seed duplicates. Including refinements and full validation, the nominal upper workload is about **280,000 local relaxations**, excluding gas-reference initialization and retries. Invalid structures can reduce the work and the eligible finalist count. No six-window hit is guaranteed.

The window slot uses a separate validity-weighted expected maximum half-window-normalized violation score from 256 reproducible latent-posterior draws. Outputs are treated as independent; observation noise is not added to these draws. It does not replace the weighted acquisition or the coverage reservation. The shortlist takes up to four mean-window hits, then conservative leaders and balanced near misses, and backfills to ten from eligible leaders. It is frozen before independent validation.

The final validation uses **20 fresh configurations and 100 attempted probes per species per finalist**. It does not share search observations. The standard 80% valid-probe threshold remains; geometries that fail contact screening are rejected observations, not zero energies. Weights, windows and contact policy are not changed adaptively during a run.

## Resume, OOM, and completion

Resubmit the **same command** after walltime expiry or failure. The launcher starts an empty seed directory, resumes its recognized checkpoint, or skips a completed search and validation before constructing a MACE evaluator. It refuses concurrent jobs for the same seed and refuses incompatible scientific settings, source or model changes.

Candidate observations and captured geometries are committed atomically to `evaluation_cache/`. An interrupted generation replays from its checkpoint, reusing completed candidate calculations with the same seed and identity. Only an unfinished/uncommitted candidate may need repeating. Do not delete the cache while resuming. The archive/checkpoint remains the authoritative search state; replayed evaluation log IDs are deduplicated and an interrupted final JSONL fragment is preserved separately.

Lower memory ceilings without changing the scientific experiment:

```bash
MAX_BATCH_ATOMS=256 MAX_BATCH_EDGES=50000 sbatch scripts/submit_big_saga.sh 123 models/s8.model
```

Autotuning is off by default to avoid raising memory use on shared GPUs. Memory ceilings partition MACE evaluations; they are different from the eight-genome acquisition batch. A single structure that cannot fit still requires a suitable GPU. Committed candidate work survives an OOM; no automatic endless retry loop is used.

Extend a compatible run without editing its source or recipe:

```bash
GENERATIONS=80 sbatch scripts/submit_big_saga.sh 123 models/s8.model
```

This continues the search and independently validates a new shortlist at the new endpoint. Previous validation results are retained in geometry/cache files but the checkpoint's validation section describes the current shortlist. A run with no eligible finalists can complete operationally; this is not a scientific success. Inspect `best`, validity rates and per-window hits.

## Results and structures

```bash
python3 scripts/summarize_campaign.py big_nh3_runs --out campaign_summary
python3 scripts/pack_results.py big_nh3_runs --out big_nh3_results.zip
```

The summary exports every current search/validation genome, its actual k, all six energies/SEs, valid fractions, rejections, window hits, and generation history. The results ZIP includes checkpoints, logs, manifests and reports, but excludes models, geometry caches and collected structures.

`collect/search` and `collect/validation` contain accepted geometries paired with clean references. Per-candidate cache EXTXYZ files also retain rejected adsorbate geometries with rejection reasons. `gas_references.json` freezes the relaxed gas energies. Keep these files for NEB selection. The search archive, not just the best candidates, should supply descriptor-stratified selections for the volcano flanks. SAGA configurations and probe placements are statistical samples, not already connected reaction pathways.

## Scope and provenance

This is a **new, self-contained campaign version** built from the available full SAGA core with the earlier literal-step fix. The local full source predates the slotting experiments; `slotting.py` is a newly implemented, tested version of the documented maximum-window-violation reservation/shortlist policy, not a byte-identical recovery of that historical module. Fresh output roots are required; do not overlay this source onto previous frozen runs.

Clean multi-start references and contact rejection are intentional production changes motivated by the geometry audit. Do not pool these energies with the old single-start, contact-unfiltered benchmark as though the protocols were identical. The new reference is the lowest sampled valid clean structure, not a proof of the global ground state. Pseudo-fcc controls, fixed bottom layers, finite slabs, sampled arrangements, regularized statistical errors and MACE model accuracy remain assumptions. The reaction windows are search heuristics, not a microkinetic activity model. In particular, the difficult NH₂ window has not been established as jointly attainable with all other windows.

## Software checks

```bash
python3 -m unittest discover -s tests -v
python3 -m saga.campaign --seed 123 --smoke-test --root smoke_runs
```

The smoke test is explicitly synthetic, short, CPU-only, and cannot validate physics. It never substitutes a toy calculator for a requested MACE run. Package checks cover slot preservation, all-atom periodic contact, clean-start selection, cache geometry recovery, deterministic interruption/resume, completed-run skipping, model/recipe guards, and literal relaxation caps. No production GPU search or new DFT validation was performed while packaging.
