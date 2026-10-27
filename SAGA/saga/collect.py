"""Geometry collection for DFT validation / active learning.

The GA already relaxes every structure it scores.  `write_candidates` keeps
the relaxed geometries of the most informative candidates (best + diverse) so
they can be re-run with a higher level of theory, and the resulting E_ads used
to fine-tune the foundation model.

Each candidate is written as a set of frames in one EXTXYZ:

    * one ``kind="slab"`` frame   — the relaxed *clean* slab;
    * several ``kind="adsorbate"`` frames — the same slab with one adsorbate,
      relaxed, tagged with the adsorbate name and the MACE-predicted ``E_ads``.

Frames share a clean-reference identity through run_id, evaluation_id,
candidate, and slab_config. The adsorbed slab atoms may relax away from their
clean-slab coordinates; pairing identifies the energy reference, not identical
coordinates. Frames are wrapped in the periodic in-plane directions on export.
"""

from __future__ import annotations

import json
import os

import numpy as np
from ase.io import write


def _info(atoms, **kv):
    atoms.info.pop("adsorbate_info", None)
    for k, v in kv.items():
        atoms.info[k] = v
    return atoms


def _add_nn(atoms):
    """Attach the MACE energy/forces as ``info['nn_energy']`` /
    ``arrays['nn_forces']`` and drop the calculator (single-point calcs bloat
    the xyz).  Frames from the batched evaluator have no calculator and may
    already carry ``nn_energy``; those are left untouched."""
    if atoms.calc is not None:
        try:
            atoms.info["nn_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["nn_forces"] = atoms.get_forces(apply_constraint=False)
        except Exception:
            pass
        atoms.calc = None
    return atoms


def _layer_planes(slab, per_layer: int) -> np.ndarray:
    """Mean z of every slab layer, assigned by atom *index*.

    The builder writes each layer with the same number of atoms, so splitting
    the index range is exact.  Clustering by z is not: an HEA plane buckles by
    up to ~1 A, which is comparable to the interlayer spacing, so the planes
    overlap in z and a cluster split would misassign atoms.
    """
    per_layer = int(per_layer or 0)
    n = len(slab)
    if per_layer < 1 or n % per_layer:
        return np.array([], dtype=float)
    n_layers = n // per_layer
    if n_layers < 2:
        return np.array([], dtype=float)
    z = np.asarray(slab.positions[:, 2], dtype=float)
    idx = np.arange(n) // per_layer
    return np.array([z[idx == j].mean() for j in range(n_layers)], dtype=float)


def _adsorbate_depth(slab, probe, per_layer: int) -> dict:
    """How far below the top plane the lowest adsorbate atom ends up.

    The adsorbate is appended after the slab, so the trailing atoms are the
    adsorbate.  A negative ``depth_below_top`` means every adsorbate atom sits
    above the top plane.

    Two flags, deliberately not the same thing:

    * ``ads_below_top`` — the lowest atom is under the plane through the top
      layer's centres.  This is *normal* for a small atom in a hollow site (H
      especially), so it is recorded, not complained about.
    * ``ads_embedded`` — the lowest atom is under the **second** layer.  That
      is not an adsorbate any more: the structure is broken, and its E_ads is
      meaningless.  This one is worth a warning.
    """
    planes = _layer_planes(slab, per_layer)
    n_ads = len(probe) - len(slab)
    if len(planes) < 2 or n_ads <= 0:
        return {}
    z_min = float(np.min(np.asarray(probe.positions[len(slab) :, 2], dtype=float)))
    return {
        "ads_z_min": z_min,
        "depth_below_top": float(planes[-1] - z_min),
        "ads_below_top": bool(z_min < planes[-1]),
        "ads_embedded": bool(z_min < planes[-2]),
    }


def _wrapped(atoms):
    """`Atoms.wrap()` mutates in place and returns None, so it cannot be
    chained into a constructor call."""
    atoms.wrap()
    return atoms


def write_candidates(collected, out_dir: str, gen: int, per_layer: int = None):
    """Write selected candidates' relaxed geometries to ``out_dir``.

    ``collected`` is a list of capture dicts with keys::

        {"id", "layers", "facet", "composite",
         "slabs": [Atoms, ...],                        # relaxed clean slabs
         "probes": [(ads, Atoms, e_ads, slab_idx), ...]}

    ``per_layer`` (``config.atoms_per_layer``) enables the structural sanity
    check on every adsorbate frame; without it the frames are still written and
    wrapped, just unannotated.

    Returns ``(path, stats)``, where ``path`` is the file written
    (``gen_####.extxyz``) or ``None`` if there was nothing to write, and
    ``stats`` counts what the sanity check saw.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    stats = {
        "n_adsorbate": 0,
        "n_below_top": 0,
        "n_embedded": 0,
        "max_depth": 0.0,
        "worst": None,
    }
    for rec in collected:
        layers_json = json.dumps(rec.get("layers"), sort_keys=True)
        composite = rec.get("composite")
        base = {
            "candidate": rec.get("id"),
            "run_id": rec.get("run_id", "unknown"),
            "evaluation_id": rec.get("evaluation_id", "unknown"),
            "gen": int(gen),
            "facet": rec.get("facet"),
            "composite": (
                round(float(composite), 6) if composite is not None else None
            ),
            "layers_json": layers_json,
        }
        slabs = rec.get("slabs", [])
        for si, slab in enumerate(slabs):
            frames.append(
                _add_nn(
                    _info(
                        _wrapped(slab.copy()), kind="slab", slab_config=int(si), **base
                    )
                )
            )
        for ads, atoms, e_ads, si in rec.get("probes", []):
            extra = {}
            # The slab of this probe, so both are measured against each other
            # rather than against a sibling configuration's slab.
            if per_layer and 0 <= int(si) < len(slabs):
                extra = _adsorbate_depth(slabs[int(si)], atoms, per_layer)
            if extra:
                stats["n_adsorbate"] += 1
                stats["n_below_top"] += int(extra["ads_below_top"])
                stats["n_embedded"] += int(extra["ads_embedded"])
                if extra["depth_below_top"] > stats["max_depth"]:
                    stats["max_depth"] = extra["depth_below_top"]
                if extra["ads_embedded"] and stats["worst"] is None:
                    stats["worst"] = f"{rec.get('id')}/{ads}" f"/cfg{int(si)}"
            frames.append(
                _add_nn(
                    _info(
                        _wrapped(atoms.copy()),
                        kind="adsorbate",
                        adsorbate=ads,
                        slab_config=int(si),
                        E_ads=float(e_ads) if e_ads is not None else None,
                        **extra,
                        **base,
                    )
                )
            )

    if not frames:
        return None, stats
    path = os.path.join(out_dir, f"gen_{int(gen):04d}.extxyz")
    write(path, frames, format="extxyz")
    if stats["n_embedded"]:
        print(
            f"  !! {stats['n_embedded']}/{stats['n_adsorbate']} collected "
            f"adsorbate frame(s) have an atom below the second layer "
            f"(deepest {stats['max_depth']:.2f} A below the top plane; "
            f"e.g. {stats['worst']}).  These are broken structures, not "
            f"adsorbates -- filter them before using them as DFT input."
        )
    return path, stats
