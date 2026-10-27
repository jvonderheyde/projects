# Release preparation checks

Executed locally during preparation with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, ASE 3.29.0, scikit-learn 1.8.0 and Matplotlib 3.10.8. Confirm the exact installed environment independently when reproducing calculations; these are software-check versions, not the historical MACE environment.

- 14 unittest checks passed, including literal optimizer caps, periodic any-adsorbate-atom contact, clean multistart selection, cache geometry round trips, deterministic interrupted-search replay, interrupted final validation, identity guards and complete-run skipping.
- The new four-adsorbate preset was exercised through synthetic launch, completion, skip and generation extension. Missing model paths fail instead of silently selecting synthetic observations.
- Synthetic CLI smoke run, repeated complete-run skip, CSV export, PNG plotting and results-only ZIP creation passed.
- Python compilation and both Slurm scripts' Bash syntax passed. A dry-run verified explicit model-file path propagation and CPU recipe settings without loading that file as MACE.
- A wheel built successfully; version and CLI imports were checked from an isolated target directory outside the source tree. Source-checkout examples remain the documented interface to the external recipe files.
- Six historical result sets were reduced to path-free numerical records; original validity flags and window-hit counts were retained. The example notebook's three code cells executed through Python/Matplotlib Agg and produced its embedded PNG. A Jupyter kernel was not available for a separate kernel-based execution.
- The results-only ZIP passed integrity checks and excluded models, XYZ files and evaluation caches.

**No new physical MACE inference, GPU search, serial/batched numerical parity test, or DFT validation was performed.** CI defines Python 3.10/3.12 checks but has not yet run on GitHub. The dependency ranges are not a claim that every allowed version has been tested. Verify one real small run with the intended model before a thesis release tag.
