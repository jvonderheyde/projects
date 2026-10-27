# Release provenance

The shared source derives from the self-contained full NH₃ campaign package prepared after the scaling-geometry audit. It includes the literal 500-step fix, periodic all-adsorbate-atom contact screen, clean multistart selection, candidate caching and resume guards.

The recovered full source predated the original slotting-experiment module. Its `slotting.py` is a new implementation of the documented coverage/window policy with software tests, **not a byte-identical recovery of the historical slotting code**. Treat this release as a new protocol and use fresh output roots. Neither the earlier toy nor its checkpoints are interchangeable with this implementation.

Release 0.3.0 adds packaging, small/medium/full presets, a launcher accepting four- and six-species ammonia recipes, Slurm submission for an explicit recipe, CSV-driven PNG plots, an example notebook, documentation and CI. The full campaign recipe remains scientifically unchanged. Random orientation, initial adsorption height and objective windows remain unchanged.

The historical example is derived from the earlier uploaded `gp_results.zip`, with exact k=2,3,5 and seeds 123/124. Its original implementation used a different sampling/validity/reference protocol. Its compact numerical data demonstrate that implementation's behavior, not the performance of release 0.3.0. Machine-specific paths and large per-probe diagnostics are omitted. The original archive remains necessary for geometry-level reproduction.

The broader historical evidence motivated current choices: four windows were often accessible; six-species joint feasibility was harder; extra NH₃ weight was not monotonically beneficial; some non-contacting N₂/NH₃ probes and reconstructed clean references required inspection. These observations justify checks and further experiments, not universal claims of catalyst activity or NNP failure.
