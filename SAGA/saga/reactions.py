"""Reaction presets: named, physically-motivated multi-step catalytic paths.

A reaction preset is fully described by four fields that everything downstream
already consumes::

    adsorbates         tuple of adsorbate names
    objectives         {adsorbate: (lo, hi)} target E_ads window (eV)
    references         {adsorbate: [(gas_species, coefficient), ...]}
    adsorbate_weights  {adsorbate: emphasis} for the weighted geometric mean

`Config.for_reaction(name)` copies a preset onto a fresh `Config`. The three
**paper-set** presets have four adsorbates each — ``nh3``, ``co2``, ``h2o`` (the
default is ``nh3``) — and each has a ``*_full`` counterpart carrying the extra
intermediates: ``nh3_full`` (6), ``co2_full`` (5), ``h2o_full`` (5).

Every window is an *absolute* electronic adsorption energy under the
computational-hydrogen-electrode (CHE) convention::

    E_ads = E(slab+ads) - E(slab) - sum_i c_i * E(gas_i)

so `references` fixes the gas reservoir for each adsorbate.  Note the windows
are **electronic** energies (0 K, no zero-point/entropy/solvation terms); the
`notes` for each preset record how they were derived from literature free-energy
diagrams / scaling relations and which are placeholders needing DFT calibration.

See ``PAPER_NOTE.md`` for the full mechanism + window rationale behind each path.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #

REACTIONS: Dict[str, dict] = {}

# A.1f — ammonia synthesis, full (adds the NH / NH2 hydrogenation steps) -------
REACTIONS["nh3_full"] = {
    "adsorbates": ("H", "N", "N2", "NH", "NH2", "NH3"),
    "objectives": {
        "H":   (-0.6, -0.2),
        "N":   (-1.2, -0.4),
        "N2":  (-0.8, -0.2),
        "NH":  (-0.2, 0.8),
        "NH2": (0.0, 1.0),
        "NH3": (-0.6, -0.1),
    },
    "references": {
        "H":   [("H2", 0.5)],
        "N":   [("N2", 0.5)],
        "N2":  [("N2", 1.0)],
        "NH":  [("N2", 0.5), ("H2", 0.5)],
        "NH2": [("N2", 0.5), ("H2", 1.0)],
        "NH3": [("NH3", 1.0)],
    },
    "adsorbate_weights": {"H": 1.0, "N": 1.0, "N2": 1.0,
                          "NH": 1.0, "NH2": 1.0, "NH3": 1.0},
    "notes": {
        "source": "original SAGA defaults; dissociative Haber-Bosch descriptor.",
        "caveat": "NH/NH2 windows are admitted guesses and are internally "
                  "inconsistent with the N window under NHx<->N scaling; kept "
                  "for backward compatibility.",
    },
}

# A.1 — ammonia synthesis, simplified (4 adsorbates; the paper set) ------------
REACTIONS["nh3"] = {
    "adsorbates": ("H", "N", "N2", "NH3"),
    "objectives": {
        "H":   (-0.6, -0.2),
        "N":   (-1.2, -0.4),
        "N2":  (-0.8, -0.2),
        "NH3": (-0.6, -0.1),
    },
    "references": {
        "H":   [("H2", 0.5)],
        "N":   [("N2", 0.5)],
        "N2":  [("N2", 1.0)],
        "NH3": [("NH3", 1.0)],
    },
    "adsorbate_weights": {"H": 1.0, "N": 1.0, "N2": 1.0, "NH3": 1.0},
    "notes": {
        "source": "physically-defensible subset of the full ammonia path: "
                  "atomic N/H binding bracket the volcano, N2 is the precursor, "
                  "NH3 is the weakly-bound product (rapid desorption).",
        "consistency": "dropping NH/NH2 removes the NHx<->N scaling "
                       "inconsistency that limited the full path.",
    },
}

# B.1 — CO2 reduction to CO (4 adsorbates; the paper set) ----------------------
REACTIONS["co2"] = {
    "adsorbates": ("H", "CO2", "COOH", "CO"),
    "objectives": {
        # Tightened windows centred on the weak-binding (noble) CO2RR->CO
        # optimum, so the search must *work* instead of trivially satisfying
        # everything at gen 0. The controlling tension is CO desorption
        # (weak CO*) vs. COOH activation, with H held weak to suppress
        # parasitic HER. Calibrated against the MACE Pareto front: with these
        # windows no candidate hits all four at generation 0, the first
        # appearing around generation 3.
        "H":    (-0.5, -0.2),
        "CO2":  (-0.5, -0.2),
        "COOH": (-0.3, 0.1),
        "CO":   (-0.5, -0.1),
    },
    "references": {
        "H":    [("H2", 0.5)],
        "CO2":  [("CO2", 1.0)],
        "COOH": [("CO2", 1.0), ("H2", 0.5)],
        "CO":   [("CO2", 1.0), ("H2", 1.0), ("H2O", -1.0)],
    },
    "adsorbate_weights": {"H": 1.0, "CO2": 1.0, "COOH": 1.0, "CO": 1.0},
    "notes": {
        "source": "CO2 + 2(H+ + e-) -> CO + H2O through the carboxyl "
                  "intermediate; CHE references account for each (H+ + e-) "
                  "pair as 0.5 H2 and the O removed to water as -H2O.",
        "symmetry": "4-adsorbate form (matches `nh3`/`h2o`) "
                    "for the paper's basic use-case set: the HER competitor H, "
                    "the reactant CO2, the RDS intermediate COOH and the "
                    "product CO. The weak-binding OH* byproduct constraint is "
                    "reserved for `co2_full`.",
        "windows": "weak CO* (desorption) balanced against near-neutral COOH* "
                   "(the RDS), with weak H*. The all-in-window region is a "
                   "Pd-rich alloy (Pd-Zn / Pd-Au / Pd-Cu, facet 110) in the "
                   "current MACE data.",
    },
}

# B.1f — CO2 reduction to CO, full (adds the weak-binding OH* byproduct) --------
REACTIONS["co2_full"] = {
    "adsorbates": ("H", "CO2", "COOH", "CO", "OH"),
    "objectives": {
        "H":    (-0.5, -0.2),
        "CO2":  (-0.5, -0.2),
        "COOH": (-0.3, 0.1),
        "CO":   (-0.5, -0.1),
        "OH":   (-0.6, -0.2),
    },
    "references": {
        "H":    [("H2", 0.5)],
        "CO2":  [("CO2", 1.0)],
        "COOH": [("CO2", 1.0), ("H2", 0.5)],
        "CO":   [("CO2", 1.0), ("H2", 1.0), ("H2O", -1.0)],
        "OH":   [("H2O", 1.0), ("H2", -0.5)],
    },
    "adsorbate_weights": {"H": 1.0, "CO2": 1.0, "COOH": 1.0,
                          "CO": 1.0, "OH": 1.0},
    "notes": {
        "source": "same CO2RR->CO mechanism and window derivation as "
                  "`co2`.",
        "caveat": "adds OH* as an explicit byproduct-over-binding constraint; "
                  "it does not enter the CO reference (that uses gas H2O).",
    },
}

# C — simplified neutral water decomposition (same public h2o preset) --------
REACTIONS["h2o"] = {
    "adsorbates": ("H", "OH", "O", "H2O"),
    "objectives": {
        "H": (-0.25, 0.05),
        "OH": (-0.6, 0.0),
        "O": (1.0, 1.6),
        "H2O": (-0.8, -0.3),
    },
    "references": {
        "H": [("H2", 0.5)],
        "OH": [("H2O", 1.0), ("H2", -0.5)],
        "O": [("H2O", 1.0), ("H2", -1.0)],
        "H2O": [("H2O", 1.0)],
    },
    "adsorbate_weights": {"H": 1.0, "OH": 1.0, "O": 1.5, "H2O": 1.0},
    "notes": {
        "source": "Neutral H2O cleavage and H/O recombination; matches the simple NEB mechanism.",
        "windows": "H/OH/O targets retained as provisional electronic screening targets, not an OER volcano. "
                   "H2O [-0.8,-0.3] eV balances molecular uptake against excessive binding; "
                   "centre -0.55 eV is near the -0.57 eV median of the supplied selected NEB minima.",
        "caveat": "H2O is referenced to gas H2O. This model-dependent window is a working guess, "
                  "not a demonstrated kinetic optimum. The selected NEB sample is biased and uses minima, "
                  "whereas SAGA scores arrangement/probe means. No voltage, solvent or pH is implied.",
    },
}

# C.f — water splitting, full (adds weak molecular H2O*) ------------------------
REACTIONS["h2o_full"] = {
    "adsorbates": ("H", "H2O", "OH", "O", "OOH"),
    "objectives": {
        "H":   (-0.25, 0.05),
        "H2O": (-0.4, 0.1),      # weak molecular H2O (solvent/reactant)
        "OH":  (-0.6, 0.0),
        "O":   (1.0, 1.6),
        "OOH": (2.6, 3.2),
    },
    "references": {
        "H":   [("H2", 0.5)],
        "H2O": [("H2O", 1.0)],
        "OH":  [("H2O", 1.0), ("H2", -0.5)],
        "O":   [("H2O", 1.0), ("H2", -1.0)],
        "OOH": [("H2O", 2.0), ("H2", -1.5)],
    },
    "adsorbate_weights": {"H": 1.0, "H2O": 1.0, "OH": 1.0,
                          "O": 1.5, "OOH": 1.5},
    "notes": {
        "source": "Legacy full OER descriptor preset; retains OOH and its historical scaling targets.",
        "caveat": "adds the weak-physisorption H2O* constraint, which is "
                  "binding in the current MACE data (best achievable E_ads(H2O) "
                  "~ -0.42 eV sits just outside the -0.4 eV edge), so no "
                  "candidate reaches all five windows simultaneously.",
    },
}


# --------------------------------------------------------------------------- #
# validation / listing
# --------------------------------------------------------------------------- #

def preset_names() -> Tuple[str, ...]:
    return tuple(REACTIONS)


def validate_preset(preset: dict) -> None:
    """Raise ``ValueError`` if a preset's four fields are mutually inconsistent.

    ``objectives``, ``references`` and ``adsorbate_weights`` must be defined for
    exactly the listed ``adsorbates`` (no missing, no extra), and every reference
    entry must be a ``(gas_species, coefficient)`` pair.
    """
    ads = tuple(preset["adsorbates"])
    for field in ("objectives", "references", "adsorbate_weights"):
        keys = set(preset[field])
        missing = [a for a in ads if a not in keys]
        extra = [k for k in keys if k not in ads]
        if missing or extra:
            raise ValueError(
                f"preset field {field!r} does not match adsorbates: "
                f"missing={missing} extra={extra}")
    for ads_name, lst in preset["references"].items():
        for entry in lst:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                    or not isinstance(entry[0], str)):
                raise ValueError(
                    f"preset reference {ads_name!r} entry {entry!r} is not a "
                    f"(gas_species, coefficient) pair")


def validate_all_presets() -> None:
    for name, preset in REACTIONS.items():
        validate_preset(preset)
