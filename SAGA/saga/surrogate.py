"""Centered heteroscedastic surrogates and deterministic composite acquisition."""

import numpy as np


def feature_scale(cfg):
    # Modest fixed depth weighting, then one learned shared length scale.
    # This avoids fitting dozens of ARD parameters to a small initial dataset.
    depth = np.linspace(0.5, 1.0, cfg.n_layers)
    return np.r_[np.repeat(depth, len(cfg.elements)), np.full(len(cfg.facets), 0.7)]


class Surrogate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = [None] * len(cfg.adsorbates)
        self.offset = np.array([sum(cfg.objectives[a]) / 2 for a in cfg.adsorbates])
        self.scale = np.ones(len(cfg.adsorbates))
        self.kernels = [None] * len(cfg.adsorbates)
        self.generation = 0

    def fit(self, X, Y, noise, generation=0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from sklearn.ensemble import RandomForestRegressor

        X = np.asarray(X, float) * feature_scale(self.cfg)
        Y = np.asarray(Y, float)
        noise = np.asarray(noise, float)
        for j in range(len(self.cfg.adsorbates)):
            mask = np.isfinite(Y[:, j])
            if not mask.any():
                self.models[j] = None
                continue
            y = Y[mask, j]
            self.offset[j] = float(y.mean())
            self.scale[j] = max(float(y.std()), 0.2)
            y = (y - self.offset[j]) / self.scale[j]
            se = np.where(
                np.isfinite(noise[mask, j]),
                noise[mask, j],
                self.cfg.arrangement_sd_prior,
            )
            alpha = (np.maximum(se, self.cfg.noise_floor) / self.scale[j]) ** 2
            if self.cfg.surrogate == "forest":
                model = RandomForestRegressor(
                    n_estimators=128,
                    min_samples_leaf=2,
                    random_state=self.cfg.seed,
                    n_jobs=-1,
                )
                model.fit(X[mask], y, sample_weight=1 / alpha)
            else:
                kernel = self.kernels[j]
                if kernel is None:
                    kernel = ConstantKernel(1.0, (0.03, 30.0)) * Matern(
                        1.0, (0.08, 8.0), nu=2.5
                    )
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    optimizer=(
                        "fmin_l_bfgs_b"
                        if self.kernels[j] is None
                        or generation % self.cfg.gp_fit_interval == 0
                        else None
                    ),
                    random_state=self.cfg.seed,
                )
                model.fit(X[mask], y)
                self.kernels[j] = model.kernel_
            self.models[j] = model
        return self

    def predict(self, X):
        X = np.asarray(X, float) * feature_scale(self.cfg)
        mu = np.tile(self.offset, (len(X), 1))
        sd = np.ones_like(mu)
        for j, model in enumerate(self.models):
            if model is None:
                continue
            if self.cfg.surrogate == "forest":
                preds = np.array([tree.predict(X) for tree in model.estimators_])
                m = preds.mean(0)
                s = preds.std(0)
                # Forest spread is heuristic, not a calibrated posterior.
                s = np.maximum(s, 0.1 / self.scale[j])
            else:
                m, s = model.predict(X, return_std=True)
            mu[:, j] = m * self.scale[j] + self.offset[j]
            sd[:, j] = s * self.scale[j]
        return mu, sd

    def kernel_state(self):
        return [None if k is None else k.theta.tolist() for k in self.kernels]

    def restore_kernels(self, state):
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        for j, theta in enumerate(state):
            if theta is not None:
                k = ConstantKernel(1.0, (0.03, 30.0)) * Matern(1.0, (0.08, 8.0), nu=2.5)
                self.kernels[j] = k.clone_with_theta(np.asarray(theta))


def acquisition(mu, sd, cfg):
    """Exact first two score moments for independent Gaussian outputs.

    For S=exp(-sum(w_i*(E_i-c_i)^2/(2*h_i^2))), E[S^r] factors into
    Gaussian integrals. This removes Monte Carlo ranking noise entirely.
    Cross-adsorbate posterior dependence is not modeled in this version.
    """
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    centers = np.array([sum(cfg.objectives[a]) / 2 for a in cfg.adsorbates])
    widths = np.array(
        [
            (cfg.objectives[a][1] - cfg.objectives[a][0]) * cfg.window_sigma / 2
            for a in cfg.adsorbates
        ]
    )
    w = np.array([cfg.adsorbate_weights[a] for a in cfg.adsorbates])
    w /= w.sum()

    def moment(r):
        t = r * w / (widths * widths)
        den = 1 + t * sd * sd
        return np.exp(
            np.sum(-0.5 * np.log(den) - 0.5 * t * (mu - centers) ** 2 / den, axis=-1)
        )

    mean = moment(1)
    second = moment(2)
    sigma = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean + cfg.kappa * sigma, mean, sigma


def distances(X, Y, cfg):
    X = np.asarray(X, float) * feature_scale(cfg)
    Y = np.asarray(Y, float) * feature_scale(cfg)
    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


def validity_prediction(X, rows, cfg):
    """Local beta-smoothed valid-probe rate; a heuristic feasibility penalty.

    Does not multiply independent per-adsorbate probabilities or claim a joint
    probability of satisfying windows. Uses the worst per-adsorbate valid rate.
    """
    if not rows:
        return np.ones(len(X))
    ref = np.array([r["features"] for r in rows])
    d = distances(X, ref, cfg)
    k = min(5, len(rows))
    near = np.argsort(d, axis=1)[:, :k]
    rates = np.array(
        [
            min(
                (r["stats"][a]["n"] + 2) / (r["stats"][a]["attempted"] + 3)
                for a in cfg.adsorbates
            )
            for r in rows
        ]
    )
    weights = np.exp(-np.take_along_axis(d, near, axis=1) ** 2 / 0.5)
    return (np.sum(weights * rates[near], axis=1) + 0.9) / (weights.sum(1) + 1)


def select_batch(X, quality, n, cfg, initial=()):
    """Greedy acquisition plus distance to already selected batch members."""
    X = np.asarray(X, float)
    quality = np.asarray(quality, float)
    if not len(X) or n <= 0:
        return []
    selected = list(dict.fromkeys(int(i) for i in initial))[:min(n,len(X))]
    available = np.ones(len(X), bool)
    if any(i<0 or i>=len(X) for i in selected): raise ValueError('invalid reserved index')
    available[selected]=False
    d = distances(X,X[selected],cfg).min(1) if selected else np.zeros(len(X))
    for _ in range(min(n, len(X))-len(selected)):
        score = quality + cfg.batch_diversity * d
        score = np.where(available, score, -np.inf)
        i = int(np.argmax(score))
        selected.append(i)
        available[i] = False
        di = distances(X, X[i : i + 1], cfg)[:, 0]
        d = di if len(selected) == 1 else np.minimum(d, di)
    return selected
