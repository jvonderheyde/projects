# Reproducibility and resume

The launcher freezes model content, Python source hashes and scientific settings per seed. Its manifest and version-3 checkpoint are run identity records. Use the same command to start, resume or skip. A path-only model rename is acceptable when bytes match. Changed windows, weights, contact checks, sampling, source or references require a fresh root.

The launcher allows compatible target-generation extensions and performance changes such as memory ceilings. CPU/GPU execution can have numerical differences despite a shared recipe; preserve environment and device details for comparisons. An identical seed is not a promise of bitwise equality across hardware or software versions.

`evaluation_cache/` stores completed candidate calculations and captured structures atomically. Resume replays an interrupted generation from the last authoritative checkpoint and reuses committed candidate results with the same identity. An uncommitted candidate can repeat. Keep the cache while resuming; deleting it removes this protection. A per-seed advisory lock prevents concurrent jobs from writing the same directory. JSONL recovery only repairs an interrupted final fragment; corruption in the middle is an error.

Independent validation can also resume. Extending the search creates a new shortlist and validation endpoint; prior structures remain available, while the current checkpoint describes the current validation stage.

To capture the environment used for thesis calculations:

```bash
python -m pip freeze > environment.txt
python -c "import platform; print(platform.platform())" > platform.txt
# On a CUDA node:
nvidia-smi > gpu.txt
```

Store these with the run outputs. The dependency ranges in `pyproject.toml` are installation bounds, not a fully tested lockfile. Historical example metadata records the environment actually used then. The current release software checks were run separately; see `RELEASE_CHECKS.md`.

Output summary CSVs are derived from the current checkpoint and include partial runs. `validation_complete` and a report indicate operational completion, not scientific success. Inspect eligibility and per-species hits. Candidate evaluations, unique genomes, relaxed structures and MACE forward passes are different work counters; do not label any one of them as another. Elapsed time is application-recorded runtime, not Slurm billed GPU time.
