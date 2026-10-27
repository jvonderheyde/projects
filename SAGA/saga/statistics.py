"""Persistent arrangement-level observations; no probe-level pseudoreplication."""

from collections import Counter
import numpy as np


def empty_observations(cfg):
    return {
        ads: {"groups": [], "attempted": 0, "rejections": {}} for ads in cfg.adsorbates
    }


def merge_observations(total, result, cfg):
    for ads in cfg.adsorbates:
        dst = total[ads]
        src = result[ads]
        # Empty groups record failed configurations; never silently erase attempts.
        groups = []
        for group in src["groups"]:
            vals = [float(v) for v in group]
            if not np.isfinite(vals).all():
                raise ValueError("nonfinite valid observation")
            groups.append(vals)
        attempted = int(src["attempted"])
        if attempted < sum(map(len, groups)):
            raise ValueError("valid samples exceed attempts")
        rejected = sum(int(n) for n in src.get("rejections", {}).values())
        if rejected != attempted - sum(map(len, groups)):
            raise ValueError("every rejected attempt must have a recorded reason")
        dst["groups"].extend(groups)
        dst["attempted"] += attempted
        dst["rejections"] = dict(
            Counter(dst["rejections"]) + Counter(src.get("rejections", {}))
        )


def summarize_ads(obs, cfg):
    groups = [np.asarray(g, float) for g in obs["groups"] if len(g)]
    n = sum(len(g) for g in groups)
    c = len(groups)
    attempted = int(obs["attempted"])
    base = {
        "mean": None,
        "se": None,
        "sd_config": None,
        "sd_probe": None,
        "n": n,
        "n_configs": c,
        "attempted_configs": len(obs["groups"]),
        "attempted": attempted,
        "valid_fraction": n / attempted if attempted else 0.0,
        "rejections": dict(obs["rejections"]),
    }
    if not groups:
        return base
    means = np.array([g.mean() for g in groups])
    sizes = np.array([len(g) for g in groups])
    # Weak variance regularization is explicit and configurable. Two prior
    # degrees of freedom prevent one arrangement from implying zero variance.
    df = sum(max(len(g) - 1, 0) for g in groups)
    ss = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    within = (ss + 2 * cfg.placement_sd_prior**2) / (df + 2)
    between_raw = (
        max(float(means.var(ddof=1)) - within * float(np.mean(1 / sizes)), 0.0)
        if c > 1
        else 0.0
    )
    between = ((c - 1) * between_raw + 2 * cfg.arrangement_sd_prior**2) / (c + 1)
    # Equal arrangement weights. For unequal surviving probe counts, the
    # placement term is sum(sigma_p^2 / P_c) / C^2.
    var_mean = between / c + within * float(np.sum(1 / sizes)) / (c * c)
    base.update(
        mean=float(means.mean()),
        se=float(np.sqrt(var_mean)),
        sd_config=float(np.sqrt(between)),
        sd_probe=float(np.sqrt(within)),
    )
    return base


def summarize(observations, cfg):
    return {ads: summarize_ads(observations[ads], cfg) for ads in cfg.adsorbates}
