"""Mean-energy objective; window feasibility is reported separately."""

import math
import numpy as np


def window_sigma_for(objective, config):
    """The Gaussian width (eV) for one adsorbate's window.

    Proportional to the window half-width, so ``config.window_sigma`` is a
    dimensionless sharpness: 1.0 puts each window edge one sigma from the
    centre (edge score ``exp(-0.5) ~ 0.61``); smaller means a sharper peak.
    """
    lo, hi = objective
    return 0.5 * (hi - lo) * config.window_sigma


def window_score(e, objective, sigma) -> float:
    """Gaussian centred on the window midpoint: ``exp(-0.5 ((e-c)/sigma)^2)``.

    The single-adsorbate factor of :func:`composite_from_energies`, which is
    their weighted geometric mean.  The composite raises this factor to the
    adsorbate's *normalized* weight, a constant exponent that cannot change
    which candidate maximizes one adsorbate's own score -- so this is the right
    key for per-adsorbate "best candidate" questions, and it keeps reporting
    helpers independent of the ranking weights.
    """
    if e is None or not np.isfinite(e):
        return 0.0
    lo, hi = objective
    center = 0.5 * (lo + hi)
    return float(np.exp(-0.5 * ((e - center) / sigma) ** 2))


def composite_from_energies(E, cfg):
    E = np.asarray(E, float)
    centers = np.array([sum(cfg.objectives[a]) / 2 for a in cfg.adsorbates])
    widths = np.array(
        [window_sigma_for(cfg.objectives[a], cfg) for a in cfg.adsorbates]
    )
    weights = np.array([cfg.adsorbate_weights[a] for a in cfg.adsorbates])
    weights /= weights.sum()
    finite = np.isfinite(E).all(axis=-1)
    z = (np.where(np.isfinite(E), E, centers) - centers) / widths
    with np.errstate(over="ignore", under="ignore"):
        score = np.exp(-0.5 * np.sum(weights * z * z, axis=-1))
    return np.where(finite, score, 0.0)


def in_window(e, window):
    return e is not None and np.isfinite(e) and window[0] <= e <= window[1]


def p_mean_in_window(mean, se, window):
    """Gaussian approximation for the latent mean, NOT a new probe/batch."""
    if mean is None or se is None or not np.isfinite([mean, se]).all():
        return None
    if se <= 0:
        return float(in_window(mean, window))
    return float(
        0.5
        * (
            math.erf((window[1] - mean) / (se * math.sqrt(2)))
            - math.erf((window[0] - mean) / (se * math.sqrt(2)))
        )
    )


def assess(stats, cfg):
    means = np.array(
        [
            stats[a]["mean"] if stats[a]["mean"] is not None else np.nan
            for a in cfg.adsorbates
        ]
    )
    ses = np.array(
        [
            stats[a]["se"] if stats[a]["se"] is not None else np.inf
            for a in cfg.adsorbates
        ]
    )
    eligible = all(
        stats[a]["n_configs"] >= cfg.min_valid_configs
        and stats[a]["valid_fraction"] >= cfg.min_valid_fraction
        for a in cfg.adsorbates
    )
    center = np.array([sum(cfg.objectives[a]) / 2 for a in cfg.adsorbates])
    worst = center + np.abs(means - center) + cfg.confidence_z * ses
    # A conservative rectangle score, not a calibrated joint confidence bound.
    conservative = (
        float(composite_from_energies(worst, cfg)) if np.isfinite(worst).all() else 0.0
    )
    robust = eligible and all(
        stats[a]["mean"] - cfg.confidence_z * stats[a]["se"] >= cfg.objectives[a][0]
        and stats[a]["mean"] + cfg.confidence_z * stats[a]["se"] <= cfg.objectives[a][1]
        for a in cfg.adsorbates
    )
    return dict(
        composite=float(composite_from_energies(means, cfg)),
        eligible=eligible,
        conservative_score=conservative if eligible else 0.0,
        all_mean_windows_met=all(
            in_window(stats[a]["mean"], cfg.objectives[a]) for a in cfg.adsorbates
        ),
        all_marginal_intervals_in_window=bool(robust),
        p_mean_in_window={
            a: p_mean_in_window(stats[a]["mean"], stats[a]["se"], cfg.objectives[a])
            for a in cfg.adsorbates
        },
    )


def ranking_key(row):
    a = row["assessment"]
    return (a["eligible"], a["conservative_score"], a["composite"])
