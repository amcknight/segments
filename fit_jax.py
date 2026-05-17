"""Fast V07 fitting on JAX: MAP via L-BFGS-B, Laplace posterior via jax.hessian.

Production fit entrypoint. Always uses padded/bucketed input internally so the
JIT cache holds a single entry per bucket size -- streaming-friendly even when
called with a single dataset.

Public API:
    warm_init_from_data(time_ms, is_died)    heuristic theta_init from data
    prewarm_buckets(buckets=...)              one-time JIT compilation
    find_map(time_ms, is_died, theta_init=None, ...)
    laplace_posterior(theta_map, time_ms, is_died)
    sample_laplace(theta_map, cov, n_samples, seed)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import config
import learning_model_v07_jax as lm


# Penalty returned by the safe wrapper when log_posterior is non-finite. Lets
# L-BFGS-B's line search detect the cliff and backtrack instead of failing on
# inf/NaN.
_NLL_PENALTY = 1e10


@jax.jit
def _neg_log_posterior(theta, time_ms_padded, is_died_padded, valid_n,
                       hl_sf_mean, hl_sf_sigma):
    """Padded, JIT-cached, safe-wrapped negative log posterior.

    hl_sf_mean, hl_sf_sigma are the (possibly pooled) halflife_sf prior
    mean and sigma_max. Pass module defaults when not pooling.
    """
    lp = lm.log_posterior_padded(theta, time_ms_padded, is_died_padded, valid_n,
                                  halflife_sf_pop_mean=hl_sf_mean,
                                  halflife_sf_pop_sigma=hl_sf_sigma)
    return jnp.where(jnp.isfinite(lp), -lp, _NLL_PENALTY)


@jax.jit
def _neg_log_posterior_value_and_grad(theta, time_ms_padded, is_died_padded, valid_n,
                                      hl_sf_mean, hl_sf_sigma):
    return jax.value_and_grad(_neg_log_posterior, argnums=0)(
        theta, time_ms_padded, is_died_padded, valid_n, hl_sf_mean, hl_sf_sigma)


@jax.jit
def _hessian(theta, time_ms_padded, is_died_padded, valid_n,
             hl_sf_mean, hl_sf_sigma):
    return jax.hessian(_neg_log_posterior, argnums=0)(
        theta, time_ms_padded, is_died_padded, valid_n, hl_sf_mean, hl_sf_sigma)


def _resolve_pool(halflife_sf_pop_mean, halflife_sf_pop_sigma):
    """Convert (None, None) -> module defaults; pass through otherwise.
    Returned values are plain Python floats; JIT wraps as scalars."""
    if halflife_sf_pop_mean is None:
        halflife_sf_pop_mean = float(lm.PRIOR_LOG_HALFLIFE_SF_MEAN)
    if halflife_sf_pop_sigma is None:
        halflife_sf_pop_sigma = float(lm.HALFLIFE_SIGMA_MAX)
    return float(halflife_sf_pop_mean), float(halflife_sf_pop_sigma)


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


def warm_init_from_data(time_ms, is_died, frac_window=0.25):
    """Heuristic theta_init from observed data. Use as the cold-start MAP init.

    Estimates per-curve asymptote and first-attempt values from late vs early
    attempt windows. Halflives stay at prior medians (the windowed data has
    no curvature signal to estimate them, and bad halflife guesses send fits
    sideways).

    Returns a length-10 numpy theta in the V07 layout.
    """
    time_ms = np.asarray(time_ms, dtype=np.float64)
    is_died = np.asarray(is_died, dtype=bool)
    N = len(time_ms)
    if N < 4:
        return np.asarray(lm.initial_theta(), dtype=np.float64)

    # bpt: small safety margin below the fastest survived time.
    survived_T = time_ms[~is_died]
    if len(survived_T) > 0:
        bpt_est = max(float(np.min(survived_T)) - 500.0, 1000.0)
    else:
        bpt_est = float(np.exp(lm.PRIOR_LOG_BPT_MEAN))

    w = max(1, int(N * frac_window))
    early, late = slice(0, w), slice(N - w, N)

    def _slop_stats(window):
        survived_in_window = time_ms[window][~is_died[window]]
        slop = survived_in_window - bpt_est
        slop = slop[slop > 0]
        if len(slop) < 2:
            return None, None
        mean_slop = float(np.mean(slop))
        std_slop  = float(np.std(slop))
        return mean_slop / bpt_est, (std_slop / mean_slop if mean_slop > 0 else 1.0)

    sf_e, ssp_e = _slop_stats(early)
    sf_l, ssp_l = _slop_stats(late)
    sf_inf  = sf_l  if sf_l  is not None else float(np.exp(lm.PRIOR_LOG_SF_INF_MEAN))
    sf_1    = sf_e  if sf_e  is not None else float(np.exp(lm.PRIOR_LOG_SF_1_MEAN))
    ssp_inf = ssp_l if ssp_l is not None else float(np.exp(lm.PRIOR_LOG_SSP_INF_MEAN))
    ssp_1   = ssp_e if ssp_e is not None else float(np.exp(lm.PRIOR_LOG_SSP_1_MEAN))

    def _alpha_from_window(window):
        died_in_window = is_died[window]
        p_die = float(died_in_window.mean()) if len(died_in_window) > 0 else 0.5
        p_die = min(max(p_die, 0.02), 0.98)
        return -np.log(1.0 - p_die)

    alpha_inf = _alpha_from_window(late)
    alpha_1   = _alpha_from_window(early)

    return np.array([
        np.log(bpt_est),
        np.log(max(sf_inf, 1e-4)),
        np.log(max(ssp_inf, 1e-3)),
        np.log(max(alpha_inf, 1e-3)),
        np.log(max(sf_1, 1e-4)),
        np.log(max(ssp_1, 1e-3)),
        np.log(max(alpha_1, 1e-3)),
        lm.PRIOR_LOG_HALFLIFE_SF_MEAN,
        lm.PRIOR_LOG_HALFLIFE_SSP_MEAN,
        lm.PRIOR_LOG_HALFLIFE_ALPHA_MEAN,
    ])


# ---------------------------------------------------------------------------
# JIT pre-warming
# ---------------------------------------------------------------------------


_PREWARMED_BUCKETS = set()


def prewarm_buckets(buckets=(64, 128, 192, 256, 320, 384, 448, 512, 768, 1024)):
    """JIT-compile the value-and-grad path for each bucket size. Call once at
    process startup so the first user-facing fits don't pay compilation cost.
    Each bucket adds ~200-400ms of one-time work."""
    theta_dummy = jnp.asarray(lm.initial_theta())
    hl_mean = jnp.asarray(float(lm.PRIOR_LOG_HALFLIFE_SF_MEAN))
    hl_sigma = jnp.asarray(float(lm.HALFLIFE_SIGMA_MAX))
    for n_max in buckets:
        if n_max in _PREWARMED_BUCKETS:
            continue
        t_dummy = jnp.full((n_max,), 30000.0, dtype=jnp.float64)
        d_dummy = jnp.zeros(n_max, dtype=bool)
        _ = float(_neg_log_posterior(theta_dummy, t_dummy, d_dummy, 1, hl_mean, hl_sigma))
        _ = _neg_log_posterior_value_and_grad(theta_dummy, t_dummy, d_dummy, 1, hl_mean, hl_sigma)
        _PREWARMED_BUCKETS.add(n_max)


# ---------------------------------------------------------------------------
# MAP fit
# ---------------------------------------------------------------------------


def _bpt_upper(time_ms, is_died):
    """Hard upper bound on log(bpt) from min observed survived time."""
    survived = np.asarray(time_ms)[~np.asarray(is_died).astype(bool)]
    if len(survived) == 0:
        return np.inf
    return float(np.log(np.min(survived)) - config.BPT_UPPER_BOUND_LOG_BACKOFF)


def find_map(time_ms, is_died, theta_init=None,
             halflife_sf_pop_mean=None, halflife_sf_pop_sigma=None,
             *, bucket_base=64, maxiter=200):
    """MAP via scipy L-BFGS-B with JAX-jitted value and gradient.

    Always bucketed: the JIT cache has one entry per bucket size, so a stream
    of fits over growing N reuses cached compilations within each bucket.

    Args:
        time_ms, is_died: numpy or jax arrays, length N.
        theta_init: starting theta. If None, uses warm_init_from_data(...).
        halflife_sf_pop_mean, halflife_sf_pop_sigma: optional empirical-Bayes
            pool overrides for the halflife_sf prior. If None, use module
            defaults. Pass the values from fit_eb_pool() to apply pooling.
        bucket_base: round N up to the nearest multiple of this for padding.
        maxiter: L-BFGS-B iteration cap.

    Returns:
        (theta_map_np, info) where info has iter, nll, converged, bucket_n.
    """
    if theta_init is None:
        theta_init = warm_init_from_data(time_ms, is_died)
    theta_init = np.asarray(theta_init, dtype=np.float64).copy()

    n_max = lm.bucket_size(len(time_ms), base=bucket_base)
    time_pad, died_pad, valid_n = lm.pad_to(time_ms, is_died, n_max)

    bpt_upper = _bpt_upper(time_ms, is_died)
    if theta_init[0] >= bpt_upper:
        theta_init[0] = bpt_upper - 0.1
    bounds = [(None, bpt_upper)] + [(None, None)] * (len(theta_init) - 1)

    hl_mean, hl_sigma = _resolve_pool(halflife_sf_pop_mean, halflife_sf_pop_sigma)
    hl_mean_j = jnp.asarray(hl_mean)
    hl_sigma_j = jnp.asarray(hl_sigma)

    def fg(theta_np):
        v, g = _neg_log_posterior_value_and_grad(
            jnp.asarray(theta_np), time_pad, died_pad, valid_n,
            hl_mean_j, hl_sigma_j)
        return float(v), np.asarray(g, dtype=np.float64)

    result = minimize(
        fg, theta_init, jac=True, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6},
    )
    info = {
        'iter': int(result.nit),
        'nll': float(result.fun),
        'converged': bool(result.success),
        'bucket_n': n_max,
        'pool_mean': hl_mean,
        'pool_sigma': hl_sigma,
    }
    return result.x, info


# ---------------------------------------------------------------------------
# Laplace posterior
# ---------------------------------------------------------------------------


def laplace_posterior(theta_map, time_ms, is_died,
                      halflife_sf_pop_mean=None, halflife_sf_pop_sigma=None,
                      *, bucket_base=64, regularize=1e-6):
    """Posterior covariance from -log_posterior Hessian at MAP.

    Pass the same pool kwargs you passed to find_map(); the Hessian must be
    consistent with the prior that produced the MAP.

    Returns (cov, ok). ok=False if H was not PD (a ridge); cov is then a
    Tikhonov-regularized pseudoinverse and the bands are approximate.
    """
    n_max = lm.bucket_size(len(time_ms), base=bucket_base)
    time_pad, died_pad, valid_n = lm.pad_to(time_ms, is_died, n_max)
    hl_mean, hl_sigma = _resolve_pool(halflife_sf_pop_mean, halflife_sf_pop_sigma)
    H = _hessian(jnp.asarray(theta_map), time_pad, died_pad, valid_n,
                 jnp.asarray(hl_mean), jnp.asarray(hl_sigma))
    H_np = np.asarray(H)
    if not np.all(np.isfinite(H_np)):
        return None, False
    try:
        eigvals = np.linalg.eigvalsh(H_np)
    except np.linalg.LinAlgError:
        return None, False
    ok = bool(np.min(eigvals) > 0)
    if not ok:
        H_np = H_np + (abs(np.min(eigvals)) + regularize) * np.eye(H_np.shape[0])
    try:
        cov = np.linalg.inv(H_np)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H_np)
        ok = False
    return cov, ok


def sample_laplace(theta_map, cov, n_samples=2000, seed=0):
    """Draw n_samples from N(theta_map, cov). Returns numpy (n_samples, 10)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(theta_map), cov, size=n_samples)
