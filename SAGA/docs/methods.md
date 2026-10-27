# Methods

## Search variables and sampling

A genome stores integer element counts in bottom-to-top slab layers and a facet. Counts sum to the number of sites in each layer. `component_scope="global"` counts unique elements across the whole slab. `component_mode="exact"` fixes that number; `"cap"` allows any number up to it. Thus capped feasible sets are nested, whereas exact-k sets differ. A larger cap need not produce a larger actual k.

Layer compositions are converted to randomly arranged slab atoms. The cell is initialized from a composition-weighted pseudo-fcc lattice constant. The release presets fix the cell and bottom layer. Changing atom ordering is part of configurational sampling, not a new genome. The core also contains additional reaction/facet options inherited from earlier work; the release recipes and checks focus on ammonia on (100)/(111).

Each polyatomic probe receives a uniform random 3D rotation and random lateral position. Its **lowest atom** starts 1.8 Å above the highest nearby slab atom (3 Å lateral radius). This is not a fixed N–metal distance: H-down NH₃ starts with N farther away than N-down NH₃. No orientation bias or reduced adsorption height has been introduced in this release.

## Energies, validity and uncertainty

For each accepted probe:

\[
E_\mathrm{ads}=E_\mathrm{slab+ads}-E_\mathrm{clean}-\sum_j\nu_j E_{j,\mathrm{gas}}.
\]

The gas coefficients and windows are defined in `saga/reactions.py`; energies are in eV per adsorbate. Clean references are relaxed separately. This energy includes any complex-induced slab reconfiguration relative to the chosen clean minimum and molecular deformation relative to the gas reference. It is not a frozen-geometry interaction energy. The lowest valid clean energy from the specified number of starts is used; this is a sampled reference, not a proven global minimum.

| Species | Reference sum | Window / eV |
|---|---|---|
| H | ½ H₂ | −0.6 to −0.2 |
| N | ½ N₂ | −1.2 to −0.4 |
| N₂ | N₂ | −0.8 to −0.2 |
| NH | ½ N₂ + ½ H₂ | −0.2 to 0.8 |
| NH₂ | ½ N₂ + H₂ | 0.0 to 1.0 |
| NH₃ | NH₃ | −0.6 to −0.1 |

`nh3` omits NH and NH₂. These windows are heuristic search targets; changing the gas reference requires changing the numerical window consistently.

Convergence, molecular integrity, distances, height and energy screens determine accepted observations. The release presets also require at least one original adsorbate atom within 1.25 times the sum of covalent radii of an original slab atom, using periodic distances. This is a screening heuristic, not a universal chemisorption criterion or proof that weak physisorption is unphysical. All-atom contact does not require N-bound NH₃ specifically. Rejection counts remain in the statistics.

Accepted probes are averaged within each configuration, then configuration means receive equal weight. Placement and arrangement variance estimates include explicit weak regularization and a noise floor. Eligibility requires configured minimum valid fractions and valid configurations. Tiny-example thresholds are deliberately sparse; robust claims require much more sampling.

## Objective and surrogate

For species i with midpoint cᵢ and half-width hᵢ (multiplied by `window_sigma`), the individual score is exp[−½((Eᵢ−cᵢ)/hᵢ)²]. Composite fitness is the weighted geometric mean of these scores, with weights normalized to sum to one. At default sharpness, the window edge has score exp(−½), approximately 0.607. Fitness therefore rewards window centers; it is not simply the number of windows hit. Eligibility, composite fitness, conservative score and explicit window hits are reported separately.

One GP per adsorbate predicts composition-mean energy and uncertainty from layer fractions and facet encoding, using a constant × Matérn-5/2 kernel and observation-dependent noise. Each fit uses the currently accumulated observations, including refinements; it is not trained only on the new proposals. Models are reconstructed on the accumulated data, while previous kernel parameters are reused; hyperparameters are optimized periodically (`gp_fit_interval`).

Acquisition uses **predicted composite-fitness mean + κ × composite-fitness standard deviation**, with κ=1 in the presets. Those moments are integrated analytically assuming independent Gaussian energy outputs. This is not an independent energy UCB for each adsorbate. A local validity estimate and diversity selection also affect selection. The optional forest uses tree spread as a heuristic uncertainty estimate, not a calibrated GP posterior.

One coverage slot can reserve an under-covered element family, and one window slot can prioritize a validity-weighted expected worst normalized window violation. The latter uses reproducible latent draws with independent outputs and no added observation noise. Slots are reservations inside the new-evaluation budget, not extra calculations beyond it. See `coverage.py`, `slotting.py`, `surrogate.py` and the selection events in `run_log.jsonl`.

Final validation uses fresh arrangements/probes for a frozen shortlist. A marginal mean interval within each window is not a calibrated joint guarantee, and composition-mean window success does not mean all states coexist on one reaction site.
