"""Structural screens independent of ASE; accepts ASE-like Atoms objects.

Thresholds are configurable heuristics. Dissociation is rejected for a molecular
adsorption descriptor by default; it is not a claim that dissociation is unphysical.
"""

import itertools
import numpy as np


def _distances(atoms):
    return np.asarray(atoms.get_all_distances(mic=True), float)


def min_distance(atoms):
    if len(atoms) < 2:
        return float("inf")
    d = _distances(atoms)
    np.fill_diagonal(d, np.inf)
    # Include a site's own neighboring periodic images as well.
    cell = np.asarray(atoms.cell)
    pbc = np.asarray(atoms.pbc, bool)
    shifts = [
        np.asarray(s) @ cell
        for s in itertools.product(*[(-1, 0, 1) if p else (0,) for p in pbc])
        if any(s)
    ]
    image_min = min((np.linalg.norm(v) for v in shifts), default=float("inf"))
    return min(float(np.min(d)), float(image_min))


def slab_reason(initial, final, converged, failed, cfg):
    if failed:
        return "slab_failed"
    if not converged:
        return "slab_unconverged"
    if (
        not np.isfinite(final.positions).all()
        or not np.isfinite(np.asarray(final.cell)).all()
    ):
        return "slab_nonfinite"
    if min_distance(final) < cfg.min_distance_ratio * min_distance(initial):
        return "slab_collision"
    old = np.linalg.norm(np.asarray(initial.cell)[:2], axis=1)
    new = np.linalg.norm(np.asarray(final.cell)[:2], axis=1)
    ratio = new / np.maximum(old, 1e-12)
    if (ratio < cfg.min_cell_ratio).any() or (ratio > cfg.max_cell_ratio).any():
        return "slab_cell_distortion"
    return None


def probe_reason(clean, probe, base, converged, failed, cfg):
    if failed:
        return "probe_failed"
    if not converged:
        return "probe_unconverged"
    if not np.isfinite(probe.positions).all():
        return "probe_nonfinite"
    n = len(clean)
    slab = probe[:n]
    if min_distance(slab) < cfg.min_distance_ratio * min_distance(clean):
        return "probe_slab_collision"
    # Use the slab coordinates in THIS relaxed probe, not the clean reference.
    z = (
        np.asarray(slab.positions)[:, 2]
        .reshape(cfg.n_layers, cfg.atoms_per_layer)
        .mean(1)
    )
    ads = np.asarray(probe.positions)[n:]
    if len(ads) != len(base):
        return "adsorbate_count"
    if ads[:, 2].min() < z[-2] - cfg.embedding_tolerance:
        return "adsorbate_embedded"
    if ads[:, 2].min() > z[-1] + cfg.max_adsorbate_height:
        return "adsorbate_desorbed"
    # An absolute near-overlap screen only; chemistry-specific bond lengths vary.
    d = _distances(probe)
    if d[n:, :n].min(initial=np.inf) < 0.5:
        return "adsorbate_collision"
    if getattr(cfg, 'require_adsorbate_contact', False):
        from ase.data import covalent_radii
        radii = covalent_radii[np.asarray(probe.numbers, int)]
        threshold = cfg.contact_radius_factor * (radii[n:,None] + radii[None,:n])
        if not np.any(d[n:,:n] <= threshold):
            return "adsorbate_no_contact"
    if cfg.molecular_integrity and len(base) > 1:
        initial = np.asarray(base.get_all_distances(mic=False))
        final = d[n:, n:]
        # Connectivity of the starting molecule using covalent radii when ASE is
        # installed. Generic fallback exists only for dependency-light tests.
        try:
            from ase.data import covalent_radii

            r = covalent_radii[np.asarray(base.numbers, int)]
            bonds = (initial < 1.25 * (r[:, None] + r[None, :])) & (initial > 0.1)
        except ImportError:
            bonds = (initial < 1.65) & (initial > 0.1)
        if np.any(final[bonds] > cfg.bond_break_ratio * initial[bonds]):
            return "adsorbate_dissociated"
        if np.any(final[bonds] < 0.5 * initial[bonds]):
            return "adsorbate_internal_collision"
    return None
