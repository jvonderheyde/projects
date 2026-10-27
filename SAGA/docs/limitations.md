# Scientific interpretation

- MACE energies/forces inherit the training domain and its errors. Good training curves do not validate every reconstructed slab or weak-binding configuration; disagreement with a foundation model does not automatically favor either model. Use targeted DFT checks.
- Pseudo-fcc starting geometries, particularly for elements with different stable phases, are structural assumptions. Clean/slab reconstruction can dominate apparent adsorption energies. Do not add the reconfiguration energy again to the reported adsorption energy.
- A high final adsorption height alone does not prove NNP failure or genuine repulsion. Orientation, weak binding, local minima, optimizer convergence and model behavior can all contribute. The contact rule changes the sampled conditional ensemble; inspect rejection fractions alongside energies.
- A mean over sites/configurations is a composition descriptor. It does not identify a single reaction pathway and cannot be combined blindly with other independently minimized states to build NEB endpoints.
- Window fitness, convex hulls of fitness profiles, flatter fitted scaling slopes and all-window hits are screening results. Narrow selection ranges and site mixtures can alter slopes without demonstrating a fundamental scaling-relation violation.
- Seed variation remains important. The earlier four-window weighting comparison improved some outcomes but did not establish seed-independent convergence. In the six-species benchmark N and NH₂ were especially difficult, motivating the full recipe's weights of two; joint feasibility remains unproven.
- SAGA provides no rate prediction here. Volcano plots need consistent connected states, activation barriers, thermochemical corrections, conditions and a microkinetic model. NEB and vibrational modules are deliberately not bundled in this SAGA-only release.

Report exact source version, model hash, recipe, seeds, validity policy, reference convention, relaxation settings and validation sampling with thesis figures. Separate results produced by different protocols.
