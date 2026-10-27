"""Integer-count genomes, with a fractional layers view for chemistry/export."""

from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass
class Candidate:
    counts: dict
    facet: int
    id: str = None

    @property
    def layers(self):
        return {
            str(l): {e: int(n) / sum(comp.values()) for e, n in comp.items() if n}
            for l, comp in self.counts.items()
        }


def count_matrix(cand, cfg):
    return np.array(
        [
            [cand.counts[str(l)].get(e, 0) for e in cfg.elements]
            for l in range(cfg.n_layers)
        ],
        dtype=int,
    )


def from_matrix(a, facet, cfg):
    a = np.asarray(a)
    if (
        a.shape != (cfg.n_layers, len(cfg.elements))
        or not np.isfinite(a).all()
        or (a < 0).any()
        or not np.equal(a, np.floor(a)).all()
        or not (a.sum(1) == cfg.atoms_per_layer).all()
    ):
        raise ValueError("invalid integer atom counts")
    if facet not in cfg.facets:
        raise ValueError("facet not in configured search space")
    k = cfg.max_components or len(cfg.elements)
    support = int((a.sum(0) > 0).sum())
    if cfg.component_mode == "exact":
        # Slab-wide and exact. Every present element already owns at least one
        # atom (integer row sums), so this feasible set is disjoint from the one
        # for any other k -- unlike the capped case, which nests.
        if support != k:
            raise ValueError(
                f"component_mode='exact' needs exactly {k} elements, found {support}"
            )
    elif (a > 0).sum(axis=1).max() > k or (
        cfg.component_scope == "global" and support > k
    ):
        raise ValueError("element support exceeds component cap")
    return Candidate(
        {
            str(l): {e: int(a[l, i]) for i, e in enumerate(cfg.elements) if a[l, i]}
            for l in range(cfg.n_layers)
        },
        int(facet),
    )


def genome_key(cand, cfg):
    return (int(cand.facet), tuple(int(n) for n in count_matrix(cand, cfg).ravel()))


def to_features(cand, cfg):
    f = np.zeros(cfg.feature_dim)
    f[: cfg.n_layers * len(cfg.elements)] = (
        count_matrix(cand, cfg).ravel() / cfg.atoms_per_layer
    )
    f[cfg.n_layers * len(cfg.elements) + cfg.facets.index(cand.facet)] = 1
    return f


def random_candidate(cfg, rng):
    n = len(cfg.elements)
    cap = min(cfg.max_components or n, n)
    a = np.zeros((cfg.n_layers, n), dtype=int)
    if cfg.component_scope == "global":
        # A slab can contain more species than a single layer has sites.
        hi = min(cap, cfg.n_layers * cfg.atoms_per_layer)
        # "cap" seeds any support up to the cap (so a run explores its whole
        # budget); "exact" always seeds the cap, so every k in a sweep is
        # genuinely a k-element alloy.
        k = hi if cfg.component_mode == "exact" else int(rng.integers(1, hi + 1))
        support = rng.choice(n, size=k, replace=False)
        for el in support:
            available = np.flatnonzero(a.sum(1) < cfg.atoms_per_layer)
            a[int(rng.choice(available)), el] += 1
        for l in range(cfg.n_layers):
            remaining = cfg.atoms_per_layer - int(a[l].sum())
            a[l, support] += rng.multinomial(remaining, rng.dirichlet(np.ones(k)))
    else:
        for l in range(cfg.n_layers):
            k = int(rng.integers(1, min(cap, cfg.atoms_per_layer) + 1))
            support = rng.choice(n, size=k, replace=False)
            a[l, support] = 1 + rng.multinomial(
                cfg.atoms_per_layer - k, rng.dirichlet(np.ones(k))
            )
    return from_matrix(a, int(rng.choice(cfg.facets)), cfg)


def enforce_exact_support(a, k, n, rng):
    """Rewrite a count matrix so its slab-wide support is exactly ``k``.

    Trims an over-full support with the *same* retention rule as the capped path
    (largest totals first, ties broken by index) and the same call sequence, so
    the two modes agree bit-for-bit whenever both are defined -- i.e. whenever
    the incoming support is already ``>= k``.  Only the top-up below is unique to
    exact mode; the capped path leaves a short support alone instead.

    Tops up with absent elements seeded from a holder that has spares.  Always
    succeeds for ``k <= a.sum()``: while the support is short of ``k`` some
    present element must hold two or more atoms (otherwise the slab would hold
    fewer than ``k`` atoms in total), so a top-up is always available.  Both
    repairs leave every per-layer atom count exact.
    """
    a = np.array(a, dtype=int, copy=True)
    totals = a.sum(0)
    present = [int(i) for i in np.flatnonzero(totals)]
    if len(present) > k:
        keep = [int(i) for i in np.argsort(-totals, kind="stable")[:k]]
        for l in range(a.shape[0]):
            removed = int(a[l].sum() - a[l, keep].sum())
            row = np.zeros(n, dtype=int)
            row[keep] = a[l, keep]
            row[keep] += rng.multinomial(removed, np.ones(len(keep)) / len(keep))
            a[l] = row
        present = [int(i) for i in np.flatnonzero(a.sum(0))]
    while len(present) < k:
        # A holder with >= 2 atoms keeps its element present after the transfer,
        # so the support grows by exactly one per iteration.
        holders = [(l, int(i)) for l in range(a.shape[0])
                   for i in np.flatnonzero(a[l]) if a[l, i] >= 2]
        l, src = holders[int(rng.integers(len(holders)))]
        add = int(rng.choice([i for i in range(n) if i not in set(present)]))
        a[l, src] -= 1
        a[l, add] += 1
        present.append(add)
    return a


def crossover(a, b, cfg, rng):
    aa = count_matrix(a, cfg)
    bb = count_matrix(b, cfg)
    use = rng.random(cfg.n_layers) < 0.5
    child = np.where(use[:, None], aa, bb)
    facet = a.facet if use[-1] else b.facet
    k = cfg.max_components or len(cfg.elements)
    if cfg.component_scope == "global" and cfg.component_mode == "exact":
        # A layer-wise mix can leave the child with fewer species than demanded,
        # so this has to add as well as remove.
        child = enforce_exact_support(child, k, len(cfg.elements), rng)
    elif cfg.component_scope == "global" and (child.sum(0) > 0).sum() > k:
        # Enforce a slab-wide support, moving removed atoms into retained species.
        support = np.argsort(-child.sum(0), kind="stable")[:k]
        for l in range(cfg.n_layers):
            removed = int(child[l].sum() - child[l, support].sum())
            row = np.zeros(len(cfg.elements), int)
            row[support] = child[l, support]
            row[support] += rng.multinomial(
                removed, np.ones(len(support)) / len(support)
            )
            child[l] = row
    return from_matrix(child, facet, cfg)


def mutate(cand, cfg, rng, max_tries=30):
    original = genome_key(cand, cfg)
    for _ in range(max_tries):
        a = count_matrix(cand, cfg)
        facet = cand.facet
        op = rng.random()
        l = int(rng.integers(cfg.n_layers))
        if op < 0.1 and len(cfg.facets) > 1:
            facet = int(rng.choice([f for f in cfg.facets if f != facet]))
        elif op < 0.3:
            # Whole-support substitution also makes global K=1 navigable.
            present = np.flatnonzero(a.sum(0))
            absent = np.flatnonzero(a.sum(0) == 0)
            if len(absent):
                src = int(rng.choice(present))
                dst = int(rng.choice(absent))
                if cfg.component_scope == "global":
                    a[:, dst] = a[:, src]
                    a[:, src] = 0
                else:
                    src = int(rng.choice(np.flatnonzero(a[l])))
                    a[l, dst] = a[l, src]
                    a[l, src] = 0
        elif op < 0.4 and cfg.n_layers > 1:
            other = int(rng.integers(cfg.n_layers))
            a[[l, other]] = a[[other, l]]
        else:
            src = int(rng.choice(np.flatnonzero(a[l])))
            dst = int(rng.integers(len(cfg.elements)))
            amount = 1 if op < 0.85 else int(rng.integers(1, a[l, src] + 1))
            a[l, src] -= amount
            a[l, dst] += amount
        try:
            child = from_matrix(a, facet, cfg)
        except ValueError:
            continue
        if genome_key(child, cfg) != original:
            return child
    return from_matrix(count_matrix(cand, cfg), cand.facet, cfg)


def candidate_dict(c):
    return {"counts": c.counts, "layers": c.layers, "facet": c.facet, "id": c.id}


def candidate_from_dict(d, cfg):
    c = Candidate(d["counts"], int(d["facet"]), d.get("id"))
    checked = from_matrix(count_matrix(c, cfg), c.facet, cfg)
    checked.id = c.id
    return checked


def frame_pair_key(info):
    # Every fresh relaxation has its own identifier, including elite remeasurement.
    return (
        info.get("run_id"),
        info.get("evaluation_id"),
        info.get("layers_json"),
        int(info.get("facet") or 0),
        int(info.get("slab_config") or 0),
    )


def frame_label(info):
    suffix = hashlib.sha256(
        json.dumps(frame_pair_key(info), sort_keys=True).encode()
    ).hexdigest()[:10]
    return f"{info.get('candidate','candidate')}@{suffix}"
