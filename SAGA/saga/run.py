"""CLI: python -m saga.run --model /path/to/model.model ..."""

import argparse
from pathlib import Path
from .config import Config
from .optimizer import ActiveLoop


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SAGA rebuilt: discrete, validity-aware alloy search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reaction",
        default="nh3",
        choices=["nh3", "nh3_full", "co2", "co2_full", "h2o", "h2o_full"],
    )
    p.add_argument("--model", dest="model_path")
    p.add_argument(
        "--demo",
        action="store_true",
        help="explicit synthetic software test; NOT chemistry",
    )
    p.add_argument("--out-dir", default="runs/saga_rebuilt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--elements", nargs="+")
    p.add_argument("--facets", type=int, nargs="+")
    p.add_argument("--slab-size", type=int, nargs=3)
    p.add_argument("--max-components", type=int)
    p.add_argument("--component-scope", choices=["global", "per_layer"])
    p.add_argument("--component-mode", choices=["cap", "exact"])
    ints = {
        "generations": "n_generations",
        "population": "population_size",
        "offspring": "offspring_size",
        "bootstrap": "bootstrap_size",
        "batch": "batch_size",
        "elite-size": "elite_size",
        "reval-elites": "reval_elites",
        "n-configs": "n_configs",
        "n-probes": "n_probes",
        "seed": "seed",
        "fix-layers": "fix_layers",
        "slab-relax-steps": "slab_relax_steps",
        "ads-relax-steps": "ads_relax_steps",
        "ref-relax-steps": "ref_relax_steps",
        "finalists": "finalist_count",
        "finalist-repeats": "finalist_repeats",
        "max-batch-atoms": "max_batch_atoms",
        "max-batch-edges": "max_batch_edges",
        "min-valid-configs": "min_valid_configs",
        "patience": "patience",
        "gp-fit-interval": "gp_fit_interval",
    }
    for flag, dest in ints.items():
        p.add_argument("--" + flag, dest=dest, type=int)
    floats = [
        "kappa",
        "mutation-rate",
        "random-fraction",
        "batch-diversity",
        "window-sigma",
        "min-valid-fraction",
        "arrangement-sd-prior",
        "placement-sd-prior",
        "noise-floor",
        "slab-relax-fmax",
        "ads-relax-fmax",
        "ref-relax-fmax",
        "embedding-tolerance",
        "max-adsorbate-height",
        "min-distance-ratio",
        "min-cell-ratio",
        "max-cell-ratio",
        "bond-break-ratio",
        "confidence-z",
        "improvement-tol",
        "maxstep",
        "nl-skin",
    ]
    for flag in floats:
        p.add_argument("--" + flag, type=float)
    p.add_argument("--energy-bounds", nargs=2, type=float)
    p.add_argument("--weights", help="ADS=WEIGHT,ADS=WEIGHT")
    p.add_argument("--device")
    p.add_argument("--precision", choices=["auto", "float32", "float64"])
    p.add_argument("--engine", choices=["auto", "batched", "ase"])
    p.add_argument("--surrogate", choices=["gp", "forest"])
    p.add_argument("--relax-algo", choices=["lbfgs", "fire"])
    for flag, dest in [
        ("no-collect", "collect"),
        ("no-relax-cell", "relax_cell"),
        ("no-autotune", "autotune"),
        ("no-cueq", "enable_cueq"),
        ("no-restrict-elements", "restrict_elements"),
        ("quiet", "progress"),
        ("allow-dissociation", "molecular_integrity"),
        ("no-coverage-explore", "coverage_explore"),
    ]:
        p.add_argument("--" + flag, dest=dest, action="store_false", default=None)
    p.add_argument(
        "--early-stop",
        action="store_true",
        default=None,
        help="optional heuristic; fixed budgets are the default",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = vars(parse_args(argv))
    reaction = args.pop("reaction")
    weights = args.pop("weights")
    if not args["demo"] and not args["model_path"]:
        raise SystemExit("--model is required unless --demo is explicitly selected")
    if args["demo"] and args["model_path"]:
        raise SystemExit("choose --demo OR --model, not both")
    overrides = {k: v for k, v in args.items() if v is not None}
    if overrides.get("model_path"):
        overrides["model_path"] = str(
            Path(overrides["model_path"]).expanduser().resolve()
        )
    # Explicit output paths are used unchanged; no implicit /K suffix.
    cfg = Config.for_reaction(reaction, **overrides)
    if weights:
        for token in weights.split(","):
            name, value = token.split("=", 1)
            cfg.adsorbate_weights[name.strip()] = float(value)
        cfg.validate()
    if cfg.demo:
        print("SYNTHETIC DEMO ONLY: output is not a catalyst prediction.", flush=True)
    report = ActiveLoop(cfg).run()
    print(f"Report: {Path(cfg.out_dir).resolve()/'report.json'}")
    if report["best"]:
        print(
            f"Validated shortlist leader: {report['best']['id']} composite={report['best']['composite']:.4f}"
        )
    else:
        print(
            "No independently validated eligible winner; inspect search_best and failure counts."
        )


if __name__ == "__main__":
    main()
