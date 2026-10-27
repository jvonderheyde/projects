"""Validated settings for the rebuilt SAGA search."""

from dataclasses import asdict, dataclass, field
from typing import Optional
import math


@dataclass
class Config:
    reaction: str = "nh3"
    elements: tuple = ("Ag", "Au", "Cu", "Fe", "Mo", "Ni", "Pd", "Pt", "Sn", "Zn")
    slab_size: tuple = (3, 3, 3)
    fix_layers: Optional[int] = None
    facets: tuple = (100, 111, 110)
    max_components: Optional[int] = None
    component_scope: str = "global"  # global slab support or per_layer support
    # "cap": max_components is an upper bound. "exact": every slab contains
    # exactly max_components elements. The two are different feasible sets --
    # capped ones nest (feasible(k) subset feasible(k+1), so the true best can
    # never fall as k rises), exact ones are disjoint per k, which is what makes
    # a composite-vs-k curve able to show a genuine interior optimum.
    component_mode: str = "cap"
    adsorbates: tuple = ("H", "N", "N2", "NH3")
    objectives: dict = field(
        default_factory=lambda: {
            "H": (-0.6, -0.2),
            "N": (-1.2, -0.4),
            "N2": (-0.8, -0.2),
            "NH3": (-0.6, -0.1),
        }
    )
    references: dict = field(
        default_factory=lambda: {
            "H": [("H2", 0.5)],
            "N": [("N2", 0.5)],
            "N2": [("N2", 1.0)],
            "NH3": [("NH3", 1.0)],
        }
    )
    adsorbate_weights: dict = field(
        default_factory=lambda: {"H": 1.0, "N": 1.0, "N2": 1.0, "NH3": 1.0}
    )
    window_sigma: float = 1.0
    objective_mode: str = "mean"  # explicit: window score of composition mean
    engine: str = "auto"
    device: str = "auto"
    precision: str = "auto"
    model_path: Optional[str] = None
    n_configs: int = 3
    n_probes: int = 5
    probe_radius: float = 3.0
    probe_height: float = 1.8
    vacuum: float = 10.0
    relax_cell: bool = True
    slab_relax_fmax: float = 0.05
    slab_relax_steps: int = 500
    ads_relax_fmax: float = 0.05
    ads_relax_steps: int = 500
    ref_relax_fmax: float = 0.01
    ref_relax_steps: int = 500
    relax_algo: str = "lbfgs"
    lbfgs_memory: int = 25
    maxstep: float = 0.2
    max_batch_atoms: int = 2144
    max_batch_edges: int = 200_000
    nl_skin: float = 0.6
    compact_every: int = 5
    enable_cueq: bool = True
    restrict_elements: bool = True
    autotune: bool = True
    cpu_workers: int = 0
    progress: bool = True
    # Structural checks are screening heuristics, not a phase-stability model.
    min_distance_ratio: float = 0.65
    min_cell_ratio: float = 0.70
    max_cell_ratio: float = 1.30
    embedding_tolerance: float = 0.20
    max_adsorbate_height: float = 5.0
    molecular_integrity: bool = True
    bond_break_ratio: float = 1.8
    energy_bounds: tuple = (-10.0, 5.0)
    min_valid_fraction: float = 0.8
    min_valid_configs: int = 2
    # Priors prevent one arrangement / identical probes giving false certainty.
    arrangement_sd_prior: float = 0.30
    placement_sd_prior: float = 0.30
    noise_floor: float = 0.02
    population_size: int = 25
    offspring_size: int = 250
    bootstrap_size: int = 24
    batch_size: int = 5
    n_generations: int = 30
    elite_size: int = 5
    reval_elites: int = 3
    mutation_rate: float = 0.5
    random_fraction: float = 0.2
    surrogate: str = "gp"
    kappa: float = 1.0
    # One batch slot per generation is reserved for element coverage, chosen
    # without reference to the acquisition score. It replaces the old
    # novelty_weight / novelty_decay pair, whose contribution was measured at
    # ~1/9th of the acquisition spread and decayed with generation.
    coverage_explore: bool = True
    window_slot: bool = False
    window_draws: int = 256
    validation_n_probes: Optional[int] = None
    require_adsorbate_contact: bool = False
    contact_radius_factor: float = 1.25
    clean_starts: int = 1
    clean_jitter: float = 0.03
    batch_diversity: float = 0.15
    gp_fit_interval: int = 5
    confidence_z: float = 1.96
    # Fixed budgets by default; final validation uses fresh arrangements.
    early_stop: bool = False
    patience: int = 15
    improvement_tol: float = 0.005
    finalist_count: int = 5
    finalist_repeats: int = 3
    collect: bool = True
    collect_dir: str = "collect"
    seed: int = 0
    out_dir: str = "runs/saga"
    resume: bool = False
    demo: bool = False

    def __post_init__(self):
        self.elements = tuple(self.elements)
        self.facets = tuple(self.facets)
        self.slab_size = tuple(self.slab_size)
        self.adsorbates = tuple(self.adsorbates)
        if self.fix_layers is None:
            self.fix_layers = 1 if self.n_layers <= 3 else 2
        self.validate()

    def validate(self):
        if len(self.slab_size) != 3 or any(
            not isinstance(v, int) or v < 1 for v in self.slab_size
        ):
            raise ValueError("slab_size must be three positive integers")
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must be nonempty and unique")
        if (
            not self.facets
            or len(set(self.facets)) != len(self.facets)
            or not set(self.facets) <= {100, 110, 111}
        ):
            raise ValueError("facets must be unique members of 100, 110, 111")
        if (
            not isinstance(self.fix_layers, int)
            or not 1 <= self.fix_layers < self.n_layers
        ):
            raise ValueError("fix_layers must leave at least one free slab layer")
        if self.max_components is not None and (
            not isinstance(self.max_components, int)
            or not 1 <= self.max_components <= len(self.elements)
        ):
            raise ValueError("max_components must be between 1 and len(elements)")
        if self.component_scope not in {"global", "per_layer"}:
            raise ValueError("component_scope must be global or per_layer")
        if self.component_mode not in {"cap", "exact"}:
            raise ValueError("component_mode must be cap or exact")
        if self.component_mode == "exact" and self.component_scope != "global":
            raise ValueError(
                "component_mode='exact' requires component_scope='global': it fixes "
                "the slab-wide element count, which is the axis the k sweep reports"
            )
        if self.component_mode == "exact" and self.max_components is None:
            raise ValueError(
                "component_mode='exact' requires max_components (unset would demand "
                "every element in every slab)"
            )
        if (
            self.component_mode == "exact"
            and self.max_components is not None
            and self.max_components > self.n_layers * self.atoms_per_layer
        ):
            raise ValueError(
                "exact component count cannot exceed the slab's atom count "
                "(each element must own at least one atom)"
            )
        for name in (
            "clean_starts",
            "window_draws",
            "n_configs",
            "n_probes",
            "population_size",
            "offspring_size",
            "bootstrap_size",
            "batch_size",
            "patience",
            "gp_fit_interval",
            "slab_relax_steps",
            "ads_relax_steps",
            "ref_relax_steps",
            "lbfgs_memory",
            "max_batch_atoms",
            "max_batch_edges",
            "min_valid_configs",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "n_generations",
            "elite_size",
            "reval_elites",
            "finalist_count",
            "finalist_repeats",
            "compact_every",
            "cpu_workers",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.elite_size > self.population_size:
            raise ValueError("elite_size must not exceed population_size")
        for name in (
            "contact_radius_factor",
            "clean_jitter",
            "window_sigma",
            "probe_radius",
            "probe_height",
            "vacuum",
            "slab_relax_fmax",
            "ads_relax_fmax",
            "ref_relax_fmax",
            "maxstep",
            "nl_skin",
            "noise_floor",
            "arrangement_sd_prior",
            "placement_sd_prior",
            "confidence_z",
            "max_adsorbate_height",
            "bond_break_ratio",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "mutation_rate",
            "random_fraction",
            "min_valid_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "kappa",
            "batch_diversity",
            "improvement_tol",
            "embedding_tolerance",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not 0 < self.min_distance_ratio < 1
            or not 0 < self.min_cell_ratio < 1 < self.max_cell_ratio
        ):
            raise ValueError("invalid geometry screening ratios")
        if (
            len(self.energy_bounds) != 2
            or not all(math.isfinite(v) for v in self.energy_bounds)
            or self.energy_bounds[0] >= self.energy_bounds[1]
        ):
            raise ValueError("invalid energy bounds")
        if self.objective_mode != "mean":
            raise ValueError(
                "this version explicitly optimizes mean adsorption energies"
            )
        if (
            self.surrogate not in {"gp", "forest"}
            or self.engine not in {"auto", "batched", "ase"}
            or self.relax_algo not in {"lbfgs", "fire"}
            or self.precision not in {"auto", "float32", "float64"}
        ):
            raise ValueError("invalid surrogate, engine, relaxer, or precision")
        if self.validation_n_probes is not None and (not isinstance(self.validation_n_probes,int) or self.validation_n_probes<1):
            raise ValueError("validation_n_probes must be a positive integer")
        self.validate_reaction()

    @property
    def atoms_per_layer(self):
        return self.slab_size[0] * self.slab_size[1]

    @property
    def n_layers(self):
        return self.slab_size[2]

    @property
    def feature_dim(self):
        return self.n_layers * len(self.elements) + len(self.facets)

    def relaxation_steps(self, stage):
        """Literal optimizer step cap, shared by ASE and batched evaluators."""
        if stage not in {"slab", "ads", "ref"}:
            raise ValueError("unknown relaxation stage: " + str(stage))
        return getattr(self, stage + "_relax_steps")

    @property
    def structures_per_generation(self):
        return (
            (self.batch_size + self.reval_elites)
            * self.n_configs
            * len(self.adsorbates)
            * self.n_probes
        )

    def relax_kwargs(self):
        return dict(
            algo=self.relax_algo,
            memory=self.lbfgs_memory,
            maxstep=self.maxstep,
            skin=self.nl_skin,
            compact_every=self.compact_every,
        )

    def to_dict(self):
        return asdict(self)

    def validate_reaction(self):
        from .reactions import validate_preset

        validate_preset(
            dict(
                adsorbates=self.adsorbates,
                objectives=self.objectives,
                references=self.references,
                adsorbate_weights=self.adsorbate_weights,
            )
        )
        if not self.adsorbates or len(set(self.adsorbates)) != len(self.adsorbates):
            raise ValueError("adsorbates must be unique and nonempty")
        for ads in self.adsorbates:
            lo, hi = self.objectives[ads]
            if (
                not all(math.isfinite(v) for v in (lo, hi))
                or lo >= hi
                or not math.isfinite(self.adsorbate_weights[ads])
                or self.adsorbate_weights[ads] <= 0
            ):
                raise ValueError(f"invalid objective/weight for {ads}")

    @classmethod
    def for_reaction(cls, name="nh3", **overrides):
        from .reactions import REACTIONS
        import copy

        preset = copy.deepcopy(REACTIONS[name])
        args = {
            k: preset[k]
            for k in ("adsorbates", "objectives", "references", "adsorbate_weights")
        }
        args.update(reaction=name)
        args.update(overrides)
        return cls(**args)
