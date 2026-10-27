"""Shared ASE/MACE sampling, structural validity, and arrangement-level outputs.

Both engines use the same construction and screening logic. A real model is
required; the separate demo evaluator is selected only with explicit --demo.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc110, fcc111, molecule
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

if __package__:
    from .geometry import build_adsorbate
    from .genome import genome_key
else:
    from geometry import build_adsorbate
    from genome import genome_key


# Pseudo-FCC reference lattice constants (Angstrom) used only to seed the
# initial cell before relaxation.
FCC_LATTICE_CONSTANTS = {
    "Ag": 4.102,
    "Al": 3.993,
    "Au": 4.137,
    "Co": 3.494,
    "Cr": 3.816,
    "Cu": 3.574,
    "Fe": 3.651,
    "Ir": 3.853,
    "Mg": 4.477,
    "Mn": 3.909,
    "Mo": 4.013,
    "Ni": 3.485,
    "Os": 3.847,
    "Pd": 3.922,
    "Pt": 3.929,
    "Re": 3.900,
    "Rh": 3.825,
    "Ru": 3.820,
    # Sn (beta-Sn is tetragonal) and Zn (hcp) have no fcc phase; these are
    # effective pseudo-fcc constants used only to seed the initial cell, which
    # is relaxed anyway.
    "Sn": 4.160,
    "Tc": 3.872,
    "Ti": 4.099,
    "V": 3.867,
    "W": 4.041,
    "Zn": 3.950,
}

_BUILDERS = {100: fcc100, 110: fcc110, 111: fcc111}


def make_calculator(config):
    """Return the MACE calculator for `config.model_path`.

    Only the serial `Evaluator` uses this; the batched path drives the raw torch
    module directly (see `engine.MaceEngine`).  There is no toy potential: the
    search is only meaningful with a trained surrogate, so a missing model is an
    error rather than a silent fallback.
    """
    if not config.model_path:
        raise ValueError("config.model_path is required (no EMT/toy fallback)")
    if config.model_path:
        import torch
        from mace.calculators import MACECalculator

        if __package__:
            from .engine import load_mace_module, resolve_devices, tune_backend
        else:
            from engine import load_mace_module, resolve_devices, tune_backend

        device = getattr(config, "device", "cpu")
        if device in (None, "", "auto"):
            try:
                device = str(resolve_devices("auto")[0])
            except Exception:
                device = "cpu"
        device = str(device).split(",")[0]
        dtype = (
            config.precision
            if config.precision != "auto"
            else ("float32" if device.startswith("cuda") else "float64")
        )
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        tune_backend([torch.device(device)], torch_dtype)
        # MACECalculator does `torch.load(map_location=device)`; loading and
        # casting on the host first is safer for float64 checkpoints.
        model = load_mace_module(config.model_path, torch_dtype).to(device)
        class PhysicalCellMACECalculator(MACECalculator):
            def _atoms_to_batch(self, atoms):
                batch = super()._atoms_to_batch(atoms)
                # MACE's neighbor builder may enlarge nonperiodic cell rows.
                # Keep its graph/shifts, but restore the physical volume used
                # by stress = -virial / volume. Nonperiodic shift indices are
                # zero, so restoring these rows does not change any neighbors.
                if not np.all(atoms.pbc) and abs(np.linalg.det(atoms.cell)) > 0:
                    cell = batch["cell"]
                    batch["cell"] = cell.new_tensor(
                        np.asarray(atoms.cell)
                    ).reshape(cell.shape)
                return batch

        return PhysicalCellMACECalculator(
            models=[model], device=device, default_dtype=dtype
        )


def composition_to_symbols(
    composition: Dict[str, float], n_sites: int, rng: np.random.Generator
) -> list:
    """Map fractional composition -> integer atom counts (largest-remainder),
    then shuffle for one configurational arrangement.
    """
    total = sum(composition.values())
    if total <= 0:
        return ["Ni"] * n_sites
    norm = {el: v / total for el, v in composition.items()}
    raw = {el: v * n_sites for el, v in norm.items()}
    base = {el: int(np.floor(v)) for el, v in raw.items()}
    rem = n_sites - sum(base.values())
    order = sorted(raw, key=lambda el: raw[el] - base[el], reverse=True)
    i = 0
    while rem > 0:
        base[order[i % len(order)]] += 1
        rem -= 1
        i += 1
    syms = []
    for el, c in base.items():
        syms.extend([el] * c)
    rng.shuffle(syms)
    return syms


def _estimate_lattice(layers: Dict[str, Dict[str, float]]) -> float:
    wsum = 0.0
    tot = 0.0
    for comp in layers.values():
        for el, f in comp.items():
            if el in FCC_LATTICE_CONSTANTS:
                wsum += f * FCC_LATTICE_CONSTANTS[el]
                tot += f
    return wsum / tot if tot > 0 else 3.70


def build_slab(
    layers: Dict[str, Dict[str, float]],
    facet: int,
    config,
    rng: np.random.Generator,
    lattice: float = None,
):
    """Build one configurational arrangement of the layered HEA slab.

    ``lattice`` (Angstrom) overrides :func:`_estimate_lattice`'s Vegard average.
    The average is the default and is accurate: measured on the FT ``k=5``
    composition it gives 3.916 A against the model's own relaxed bulk 3.932 A
    (0.4%), so no relaxation is spent improving on it.
    """
    a = lattice if lattice is not None else _estimate_lattice(layers)
    slab = _BUILDERS[facet]("Cu", size=config.slab_size, a=a, vacuum=config.vacuum)
    missing = [str(l) for l in range(config.n_layers) if str(l) not in layers]
    if missing:
        raise ValueError(
            f"slab_size asks for {config.n_layers} layers but the composition "
            f"only describes {sorted(layers)} (missing {missing}). The SAGA run "
            f"this came from used a different slab thickness -- rerun SAGA with "
            f"--layers {config.n_layers}, or set --slab-size to match the "
            f"checkpoint (its collect/ slabs record their own thickness)."
        )
    syms = []
    for l in range(config.n_layers):
        syms.extend(composition_to_symbols(layers[str(l)], config.atoms_per_layer, rng))
    slab.set_chemical_symbols(syms)

    # Pin the bottom `fix_layers` layers. The ASE fcc builders order atoms
    # bottom-plane-first with exactly `atoms_per_layer` atoms per composition
    # layer, so the bottom `fix_layers` layers are simply the first
    # `fix_layers * atoms_per_layer` indices. Fixing more than one layer matters
    # for thick slabs (4+ layers): pinning only the bottom lets the adsorbate
    # pull the layers above it up and separate them from the bulk, which a
    # 3-layer slab happens to hide because the pull is balanced by the fixed
    # bottom. Set `fix_layers=2` when going to 4 or more layers.
    fix = getattr(config, "fix_layers", None)
    if fix is None:  # e.g. slab_size mutated after construction
        fix = 1 if config.n_layers <= 3 else 2
    fix = max(1, min(int(fix), config.n_layers))
    mask = [i < fix * config.atoms_per_layer for i in range(len(slab))]
    slab.set_constraint(FixAtoms(mask=mask))
    return slab


def place_probe(slab, base, config, rng: np.random.Generator):
    """Drop one randomly oriented adsorbate onto a random surface site.

    Shared by both evaluators so the sampled starting geometries — and hence
    the physics — are identical whichever driver runs the relaxation.
    """
    p = slab.copy()
    u, v = rng.uniform(0, 1, 2)
    cell = p.get_cell()
    pos_xy = u * cell[0] + v * cell[1]

    from ase.geometry import find_mic

    delta = p.positions - pos_xy
    delta[:, 2] = 0.0
    delta, _ = find_mic(delta, cell, pbc=(True, True, False))
    nearby = np.linalg.norm(delta[:, :2], axis=1) < config.probe_radius
    z_ref = (
        np.max(p.positions[nearby, 2]) if nearby.any() else np.max(p.positions[:, 2])
    )

    ads = base.copy()
    if len(ads) > 1:
        from scipy.spatial.transform import Rotation

        rotation = Rotation.random(random_state=rng).as_matrix()
        center = ads.positions.mean(0)
        ads.positions[:] = (ads.positions - center) @ rotation.T + center

    # Anchor the LOWEST atom at z_ref + probe_height so rotated molecules
    # never start clipped into the surface.
    pos = ads.get_positions()
    shift = pos_xy[:2] - pos[:, :2].mean(axis=0)
    zshift = (z_ref + config.probe_height) - pos[:, 2].min()
    ads.translate([shift[0], shift[1], zshift])
    p += ads
    return p


def required_atomic_numbers(config) -> List[int]:
    """Every element this configuration can ever put into a structure.

    Used to strip a foundation model down to the elements in play; missing one
    would be a hard error at evaluation time, so this deliberately unions the
    alloy elements with the adsorbates *and* their gas-phase references.
    """
    from ase.data import atomic_numbers as Z

    zs = {int(Z[el]) for el in config.elements}
    gas_species = {s for lst in config.references.values() for s, _ in lst}
    for sp in gas_species:
        zs.update(int(z) for z in molecule(sp).get_atomic_numbers())
    # Adsorbates go through build_adsorbate so radical intermediates (COOH,
    # OOH, ...) with no G2 entry still contribute their atomic numbers.
    for ads in config.adsorbates:
        zs.update(int(z) for z in build_adsorbate(ads).get_atomic_numbers())
    return sorted(zs)


def reference_molecule(symbol: str):
    """A gas-phase reference molecule in a large box (no periodic images)."""
    at = molecule(symbol)
    at.cell = [15, 15, 15]
    at.pbc = False
    at.center()
    return at


from .statistics import empty_observations
from .validation import slab_reason, probe_reason


class Evaluator:
    """Shared sampling, validity, and aggregation logic for both engines.

    Results retain one list per attempted arrangement for each adsorbate.
    Candidate caching belongs to the optimizer archive; every call here is fresh.
    """

    def __init__(self, config):
        self.config = config
        self.calc = make_calculator(config)
        self.captured = {}
        self.stats = {
            "slab_relax": 0,
            "probe_relax": 0,
            "reference_relax": 0,
            "forwards": 0,
        }
        self._refs = self._reference_energies()

    def _relax_many(self, atoms, stage):
        cfg = self.config
        fmax = getattr(cfg, stage + "_relax_fmax")
        steps = cfg.relaxation_steps(stage)
        out = []
        for initial in atoms:
            final = initial.copy()
            final.calc = self.calc
            target = (
                FrechetCellFilter(final, mask=[1, 1, 0, 0, 0, 0])
                if stage == "slab" and cfg.relax_cell
                else final
            )
            if cfg.relax_algo == "fire":
                from ase.optimize import FIRE

                opt = FIRE(target, logfile=None, maxstep=cfg.maxstep)
            else:
                opt = LBFGS(
                    target, logfile=None, maxstep=cfg.maxstep, memory=cfg.lbfgs_memory
                )
            converged = bool(opt.run(fmax=fmax, steps=steps))
            energy = float(final.get_potential_energy())
            forces = final.get_forces(apply_constraint=False)
            failed = (
                not np.isfinite(energy)
                or not np.isfinite(forces).all()
                or np.linalg.norm(forces, axis=1).max() > 100.0
            )
            final.info["nn_energy"] = energy
            final.arrays["nn_forces"] = forces
            out.append((final, energy, converged, failed))
        self.stats[
            {"slab": "slab_relax", "ads": "probe_relax", "ref": "reference_relax"}[
                stage
            ]
        ] += len(atoms)
        return out

    def _reference_energies(self):
        species = sorted(
            {s for refs in self.config.references.values() for s, _ in refs}
        )
        results = self._relax_many([reference_molecule(s) for s in species], "ref")
        energies = {}
        for species_name, (_, e, conv, failed) in zip(species, results):
            if failed or not conv or not np.isfinite(e):
                raise RuntimeError(
                    f"Gas reference {species_name} did not converge; inspect model and reference settings"
                )
            energies[species_name] = e
        self.gas_energies = energies
        return {
            ads: sum(c * energies[s] for s, c in refs)
            for ads, refs in self.config.references.items()
        }

    def evaluate_many(self, cands, rng, label="", force=None):
        cfg = self.config
        self.captured = {}
        self.last_cached = [False] * len(cands)
        out = [empty_observations(cfg) for _ in cands]
        for result in out:
            for ads in cfg.adsorbates:
                result[ads]["groups"] = [[] for _ in range(cfg.n_configs)]
                result[ads]["attempted"] = cfg.n_configs * cfg.n_probes
        seeds = rng.integers(0, 2**62, size=len(cands))
        streams = [np.random.default_rng(int(s)) for s in seeds]
        captures = [
            dict(id=c.id, layers=c.layers, facet=c.facet, slabs=[], probes=[], rejected=[])
            for c in cands
        ]

        def reject(i, ads, reason, n=1):
            d = out[i][ads]["rejections"]
            d[reason] = d.get(reason, 0) + n

        slabs = []
        meta = []
        for i, c in enumerate(cands):
            for j in range(cfg.n_configs):
                slab = build_slab(c.layers, c.facet, cfg, streams[i])
                slabs.append(slab)
                meta.append((i, j))
        starts=[]; owners=[]
        for k,(slab,(i,j)) in enumerate(zip(slabs,meta)):
            for attempt in range(cfg.clean_starts):
                start=slab.copy()
                if attempt:
                    fixed=np.zeros(len(start),bool)
                    for constraint in start.constraints:
                        if hasattr(constraint,'get_indices'): fixed[constraint.get_indices()]=True
                    delta=streams[i].normal(0,cfg.clean_jitter,(len(start),3)); delta[fixed]=0
                    start.positions[:]+=delta
                starts.append(start);owners.append(k)
        start_results=self._relax_many(starts,"slab")
        relaxed=[None]*len(slabs)
        for k,start,result in zip(owners,starts,start_results):
            clean,e,conv,failed=result
            if slab_reason(slabs[k],clean,conv,failed or not np.isfinite(e),cfg) is None:
                if relaxed[k] is None or e<relaxed[k][1]: relaxed[k]=result
        for k,result in enumerate(relaxed):
            if result is None: relaxed[k]=(slabs[k].copy(),float('nan'),False,True)
            else:
                result[0].info['clean_reference_starts']=cfg.clean_starts
                result[0].info['clean_reference_jitter_A']=cfg.clean_jitter
        probes = []
        pm = []
        bases = {a: build_adsorbate(a) for a in cfg.adsorbates}
        for k, ((i, j), (clean, e, conv, failed)) in enumerate(zip(meta, relaxed)):
            reason = slab_reason(
                slabs[k], clean, conv, failed or not np.isfinite(e), cfg
            )
            if reason:
                for ads in cfg.adsorbates:
                    reject(i, ads, reason, cfg.n_probes)
                continue
            si = len(captures[i]["slabs"])
            clean.info.update(candidate=cands[i].id,config_index=j)
            captures[i]["slabs"].append(clean.copy())
            for ads in cfg.adsorbates:
                for _ in range(cfg.n_probes):
                    probes.append(place_probe(clean, bases[ads], cfg, streams[i]))
                    pm.append((i, j, k, si, ads))
        results = self._relax_many(probes, "ads")
        for (i, j, k, si, ads), (probe, e, conv, failed) in zip(pm, results):
            clean, eclean, _, _ = relaxed[k]
            reason = probe_reason(
                clean, probe, bases[ads], conv, failed or not np.isfinite(e), cfg
            )
            eads = float(e - eclean - self._refs[ads])
            if (
                reason is None
                and not cfg.energy_bounds[0] <= eads <= cfg.energy_bounds[1]
            ):
                reason = "energy_bounds"
            if reason:
                reject(i, ads, reason)
                if cfg.collect:
                    probe.info.update(adsorbate=ads, rejection=reason, E_ads=eads, slab_config=si, config_index=j)
                    captures[i]['rejected'].append(probe.copy())
                continue
            probe.info.update(adsorbate=ads, E_ads=eads, slab_config=si, config_index=j)
            out[i][ads]["groups"][j].append(eads)
            captures[i]["probes"].append((ads, probe.copy(), eads, si))
        if cfg.collect:
            self.captured = {c.id: cap for c, cap in zip(cands, captures)}
        return out

    def cached_keys(self):
        return set()

    def evaluate(self, cand, rng):
        return self.evaluate_many([cand], rng)[0]


class BatchedEvaluator(Evaluator):
    """The original batched MACE engine behind the shared evaluator contract."""

    def __init__(self, config):
        from .engine import make_engine, describe_engine

        self.config = config
        self.engine = make_engine(config, keep_z=required_atomic_numbers(config))
        self.description = describe_engine(self.engine)
        if config.relax_cell and not self.engine.supports_virial:
            raise RuntimeError("Cell relaxation requested but engine has no virial")
        self.captured = {}
        self.stats = {
            "slab_relax": 0,
            "probe_relax": 0,
            "reference_relax": 0,
            "forwards": 0,
        }
        if config.autotune:
            from .engine import Sys, autotune_batch_atoms
            from .genome import random_candidate

            rng = np.random.default_rng(config.seed)
            samples = []
            for _ in range(4):
                c = random_candidate(config, rng)
                slab = build_slab(c.layers, c.facet, config, rng)
                samples.append(
                    Sys.from_atoms(
                        place_probe(
                            slab, build_adsorbate(config.adsorbates[-1]), config, rng
                        )
                    )
                )
            autotune_batch_atoms(
                self.engine,
                samples,
                need_virial=config.relax_cell,
                verbose=config.progress,
            )
        self._refs = self._reference_energies()

    def _relax_many(self, atoms, stage):
        from .engine import Sys
        from .relax import relax

        cfg = self.config
        steps = cfg.relaxation_steps(stage)
        systems = [Sys.from_atoms(a) for a in atoms]
        result = relax(
            self.engine,
            systems,
            fmax=getattr(cfg, stage + "_relax_fmax"),
            steps=steps,
            cell_mask=(
                (True, True, False) if stage == "slab" and cfg.relax_cell else None
            ),
            **cfg.relax_kwargs(),
        )
        self.stats[
            {"slab": "slab_relax", "ads": "probe_relax", "ref": "reference_relax"}[
                stage
            ]
        ] += len(atoms)
        self.stats["forwards"] += result.n_forward
        out = []
        for i, initial in enumerate(atoms):
            final = initial.copy()
            final.set_cell(result.cells[i])
            final.set_positions(result.positions[i], apply_constraint=False)
            final.info["nn_energy"] = float(result.energy[i])
            out.append(
                (
                    final,
                    float(result.energy[i]),
                    bool(result.converged[i]),
                    bool(result.failed[i]),
                )
            )
        return out


def make_evaluator(config):
    if not config.model_path:
        raise ValueError("--model is required for real evaluation")
    return Evaluator(config) if config.engine == "ase" else BatchedEvaluator(config)
