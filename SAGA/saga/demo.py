"""Synthetic evaluator for software tests only; never used as a MACE fallback."""

import numpy as np
from .genome import count_matrix
from .statistics import empty_observations


class DemoEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.captured = {}
        self.stats = {"candidate_evaluations": 0}

    def evaluate_many(self, cands, rng, label="", force=None):
        cfg = self.cfg
        out = []
        # Same deterministic landscape for every run seed; only sampling varies.
        weights = np.random.default_rng(9182).normal(
            0, 0.8, (len(cfg.adsorbates), cfg.n_layers, len(cfg.elements))
        )
        for c in cands:
            fraction = count_matrix(c, cfg) / cfg.atoms_per_layer
            result = empty_observations(cfg)
            arrangement = rng.normal(0, 0.10, (cfg.n_configs, len(cfg.adsorbates)))
            for j, ads in enumerate(cfg.adsorbates):
                center = sum(cfg.objectives[ads]) / 2
                energy = (
                    center
                    + np.sum(weights[j] * fraction) / cfg.n_layers
                    + 0.1 * (c.facet == 100)
                )
                result[ads]["groups"] = [
                    (
                        energy + arrangement[k, j] + rng.normal(0, 0.12, cfg.n_probes)
                    ).tolist()
                    for k in range(cfg.n_configs)
                ]
                result[ads]["attempted"] = cfg.n_configs * cfg.n_probes
            out.append(result)
        self.stats["candidate_evaluations"] += len(cands)
        return out
