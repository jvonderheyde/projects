# Development

Use an editable installation and run `python -m unittest discover -s tests -v` from the repository root. The tests use synthetic observations or explicit calculator/test doubles and require no private MACE model. Physical MACE execution is separate.

Keep the tiny and full recipes on the same core. Put protocol choices in recipes, not duplicated search implementations. Preserve literal relaxation caps, model identity, explicit validity/rejection counts, configuration-level statistics and interruption-safe cache handling. Scientific changes need a fresh output root and a provenance note.

Add meaningful regression tests when altering acquisition, validity or persistence. Compare fixed-geometry energies/forces for serial/batched engine changes using a trusted compatible MACE model before using changed engines for production. Include actual package/hardware versions with a bug report and a minimal reproducible example; omit private model weights unless redistribution is authorized.

Implementation map: `config.py` validates settings; `genome.py` handles genomes/operators; `chemistry.py` constructs and evaluates structures; `engine.py`/`relax.py` provide batched inference/optimization; `statistics.py` summarizes grouped samples; `objective.py` scores them; `surrogate.py`, `coverage.py`, `slotting.py` select evaluations; `optimizer.py` controls search/validation; `campaign.py` handles launch/identity/locking; `evaluation_cache.py` commits candidate work. `run.py` is an inherited general CLI; the public examples use `campaign.py` for consistent automatic resume.
