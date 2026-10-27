"""Explicit adsorbate geometries for species ASE's G2 database cannot build.

`ase.build.molecule` (the G2 set) covers the common closed-shell gases and a
handful of radicals, but it does **not** include the carboxyl radical (``COOH``)
or hydroperoxyl (``OOH``), and it spells several others under different names
(``CHO`` -> ``HCO``, ``CH2O`` -> ``H2CO``, ``CH2OH`` -> ``H2COH``).

`build_adsorbate` resolves every reaction adsorbate to a concrete :class:`~ase.Atoms`:

* an explicit geometry when G2 has none (``COOH``, ``OOH``);
* the G2 alias otherwise (``CHO``, ``CH2O``, ``CH2OH``);
* plain ``ase.build.molecule`` as the final fallback.

The explicit geometries are chemically sensible gas-phase-like starting points;
they are randomly rotated and then relaxed on the slab, so exact bond angles are
not critical — what matters is the correct atoms and no overlapping positions.
"""

from __future__ import annotations

from ase import Atoms
from ase.build import molecule

# Radical / polyatomic adsorbates NOT present in the G2 database, given as
# explicit ASE Atoms (reasonable gas-phase-like geometry).
_EXPLICIT = {
    # Carboxyl radical HOCO (trans conformer). C=O ~1.18 A, C-OH ~1.34 A,
    # O-H ~0.97 A, O-C-O ~127 deg.
    "COOH": Atoms(
        symbols="COOH",
        positions=[
            (0.000, 0.000, 0.000),   # C
            (1.175, 0.000, 0.000),   # O  (carbonyl, C=O)
            (-0.806, 1.070, 0.000),  # O  (hydroxyl, C-OH)
            (-0.250, 1.940, 0.000),  # H  (on the hydroxyl O)
        ]),
    # Hydroperoxyl radical HOO. O-O ~1.33 A, O-H ~0.97 A, O-O-H ~104 deg.
    "OOH": Atoms(
        symbols="OOH",
        positions=[
            (0.000, 0.000, 0.000),   # O
            (1.330, 0.000, 0.000),   # O
            (1.565, 0.941, 0.000),   # H
        ]),
}

# G2 spellings for adsorbates SAGA refers to by their chemical name.
_G2_ALIASES = {
    "CHO":   "HCO",    # formyl radical
    "CH2O":  "H2CO",   # formaldehyde
    "CH2OH": "H2COH",  # hydroxymethyl radical
    # CH3O (methoxy) is already a G2 name; listed for documentation.
    "CH3O":  "CH3O",
}


def build_adsorbate(name: str) -> Atoms:
    """Build one adsorbate as an :class:`~ase.Atoms` (explicit geometry when
    G2 lacks the species, else G2 alias, else plain ``molecule``)."""
    if name in _EXPLICIT:
        return _EXPLICIT[name].copy()
    g2 = _G2_ALIASES.get(name, name)
    return molecule(g2)


def radical_adsorbates() -> tuple:
    """Names of radical adsorbates that need special handling."""
    return tuple(_EXPLICIT) + tuple(_G2_ALIASES)
