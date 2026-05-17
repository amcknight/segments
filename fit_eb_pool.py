"""Empirical-Bayes pooling for halflife_sf across many segments.

The lightest principled HYPER step (see external_docs/speed_and_techniques.md).
Point-estimates the population (mean, sigma) for log(halflife_sf), then refits
each segment with that as its prior. Iterates until convergence.

Standard scaling pattern for production hierarchical models: per-segment fits
remain independent and fully parallel, conditional on a small shared block of
point-estimated hyperparameters. To upgrade to fully Bayesian later, swap
this loop for NUTS over the joint hyper + per-segment posteriors.

Public API:
    Pool: namedtuple holding (log_halflife_sf_mean, log_halflife_sf_sigma).
    fit_eb_pool(segments, ...)         iterate EB until convergence.
    fit_independent(segments, ...)     baseline: each segment fit alone.
"""
from typing import NamedTuple

import numpy as np

import learning_model_v07 as lm
import fit_jax


class Pool(NamedTuple):
    """Empirical-Bayes pool: per-segment prior on log(halflife_sf)."""
    log_halflife_sf_mean: float
    log_halflife_sf_sigma: float

    @classmethod
    def from_defaults(cls):
        """Pool == module defaults (no pooling)."""
        return cls(float(lm.PRIOR_LOG_HALFLIFE_SF_MEAN),
                   float(lm.HALFLIFE_SIGMA_MAX))


def _fit_one(seg, pool, theta_init=None):
    """Fit one segment under the given pool prior. Returns (theta_map, info)."""
    import learning_model_v07_jax as lm_jx
    time_ms, is_died = lm_jx.data_to_arrays(seg['data'])
    return fit_jax.find_map(
        np.asarray(time_ms), np.asarray(is_died),
        theta_init=theta_init,
        halflife_sf_pop_mean=pool.log_halflife_sf_mean,
        halflife_sf_pop_sigma=pool.log_halflife_sf_sigma,
    )


def fit_independent(segments):
    """Fit each segment with module-default priors (no pooling). Baseline."""
    pool = Pool.from_defaults()
    maps, infos = [], []
    for seg in segments:
        theta_map, info = _fit_one(seg, pool)
        maps.append(theta_map)
        infos.append(info)
    return maps, infos


def fit_eb_pool(segments, max_iter=10, tol=1e-3, verbose=False,
                sigma_floor=0.05):
    """Iterate the EB loop on halflife_sf.

    Each iteration:
        1. Fit each segment with the current pool prior.
        2. Re-estimate (mu_pop, sigma_pop) from the S MAP log_halflife_sf values.
        3. Stop when both move less than `tol` in log-space.

    Args:
        segments: list of {'data': [...]} dicts, as from generate_multi_segment.
        max_iter: cap on EB iterations (typically converges in 3-8).
        tol: convergence threshold in log-space for both mean and sigma.
        sigma_floor: don't let sigma_pop shrink below this (defensive against
            a one-segment dataset or degenerate variance).
        verbose: print one line per iteration.

    Returns:
        (pool, maps, history) where:
            pool: final Pool
            maps: list of per-segment MAP thetas under the final pool
            history: list of Pool values across iterations (for debugging)
    """
    pool = Pool.from_defaults()
    history = [pool]
    prev_maps = None

    for it in range(max_iter):
        # Warm-start each segment from its previous MAP (faster convergence).
        maps = []
        for s_idx, seg in enumerate(segments):
            init = prev_maps[s_idx] if prev_maps else None
            theta_map, _ = _fit_one(seg, pool, theta_init=init)
            maps.append(theta_map)
        prev_maps = maps

        # Re-estimate pool from segment MAPs.
        halflife_logs = np.array([t[lm.IDX_LOG_HALFLIFE_SF] for t in maps])
        new_mean = float(halflife_logs.mean())
        new_sigma = max(float(halflife_logs.std()), sigma_floor)
        new_pool = Pool(new_mean, new_sigma)

        d_mean = abs(new_pool.log_halflife_sf_mean - pool.log_halflife_sf_mean)
        d_sigma = abs(new_pool.log_halflife_sf_sigma - pool.log_halflife_sf_sigma)
        history.append(new_pool)
        if verbose:
            print(f"  EB iter {it+1}: pool=({new_pool.log_halflife_sf_mean:+.3f}, "
                  f"{new_pool.log_halflife_sf_sigma:.3f})  "
                  f"natural halflife median={np.exp(new_pool.log_halflife_sf_mean):.1f}  "
                  f"|Δμ|={d_mean:.4f}  |Δσ|={d_sigma:.4f}")
        pool = new_pool
        if d_mean < tol and d_sigma < tol:
            break

    return pool, maps, history
