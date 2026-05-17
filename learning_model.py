"""Learning-curve extension to the static segment model.

The static model fits each segment with time-invariant params. This module lets
the skill-dependent params evolve over attempts within a segment, per
external_docs/learning_curves.md:

    log theta(n) = log theta_inf + Delta * exp(-n / tau)

v1 commitment: hazard model is haz1 (single bin, K=1 PiecewiseConstantHazard).
slop_frac, slop_spread, and alpha all get learning curves. bpt stays static.

theta layout (length 10):
    theta[0] = log(bpt)               -- static
    theta[1] = log(slop_frac_inf)     -- asymptote
    theta[2] = log(slop_spread_inf)   -- asymptote
    theta[3] = log(alpha_inf)         -- asymptote (the single haz1 param)
    theta[4] = Delta_sf
    theta[5] = Delta_ssp
    theta[6] = Delta_alpha
    theta[7] = log(tau_sf)
    theta[8] = log(tau_ssp)
    theta[9] = log(tau_alpha)

The first 4 entries match the static model's theta exactly with K=1 haz1, so at
each attempt n we build an effective static-theta by substituting the curve
values into entries 1, 2, 3 and reuse model.py's per-attempt likelihood
primitives unchanged.
"""
import numpy as np
from scipy.stats import norm

import config
import model
from hazard import PiecewiseConstantHazard


N_STATIC = 4         # log(bpt), log(sf_inf), log(ssp_inf), log(alpha_inf)
N_LEARNING = 3       # sf, ssp, alpha all learn
N_PARAMS = N_STATIC + 2 * N_LEARNING   # 10


# Learning-curve priors (see learning_curves.md "Priors (v1)")
PRIOR_DELTA_MEAN = 0.0
PRIOR_DELTA_SD = 2.0
PRIOR_LOG_TAU_MEAN = np.log(30.0)
PRIOR_LOG_TAU_SD = 1.5


def haz1():
    """The committed v1 hazard: single-bin piecewise-constant."""
    return PiecewiseConstantHazard(K=1, bin_edges=[0.0, 1.0])


def learning_curve(log_theta_inf, delta, log_tau, n):
    """log theta(n) = log theta_inf + delta * exp(-n / tau).

    Vectorized over n (n may be scalar or array). log_theta_inf, delta, log_tau
    are scalars (one learning param).
    """
    tau = np.exp(log_tau)
    return log_theta_inf + delta * np.exp(-np.asarray(n, dtype=float) / tau)


def static_theta_at_n(theta, n):
    """Effective static-model theta at attempt number n (1-indexed).

    Substitutes the learning-curve values into theta[1:4]. Returns a length-4
    vector laid out as model.unpack expects for haz1 (bpt, sf, ssp, alpha).
    """
    asymp = theta[:N_STATIC]            # [log_bpt, log_sf_inf, log_ssp_inf, log_alpha_inf]
    deltas = theta[N_STATIC:N_STATIC + N_LEARNING]
    log_taus = theta[N_STATIC + N_LEARNING:N_PARAMS]
    taus = np.exp(log_taus)
    decay = np.exp(-float(n) / taus)
    out = asymp.copy()
    out[1:1 + N_LEARNING] = asymp[1:1 + N_LEARNING] + deltas * decay
    return out


def log_likelihood(theta, data, hazard):
    """Sum of per-attempt log-likelihoods with theta(n) substituted each attempt.

    Reuses model.log_lik_survived / model.log_lik_died unchanged; only the
    theta passed to them varies per attempt.
    """
    total = 0.0
    for n, (outcome, time_ms) in enumerate(data, start=1):
        theta_n = static_theta_at_n(theta, n)
        bpt, sf, ssp, haz_params = model.unpack(theta_n, hazard)
        mu_log, sigma_log = model.lognormal_params(sf, ssp, bpt)
        if outcome == 'survived':
            total += model.log_lik_survived(time_ms, bpt, mu_log, sigma_log,
                                            hazard, haz_params)
        else:
            tau_obs = time_ms - config.RESPAWN_MS
            total += model.log_lik_died(tau_obs, bpt, mu_log, sigma_log,
                                        hazard, haz_params)
        if not np.isfinite(total):
            return -np.inf
    return total


def log_prior(theta, hazard):
    """Prior factorizes: asymptotes inherit the static priors; (Delta, log tau)
    use the learning-curve priors."""
    lp = 0.0
    # Asymptotes -- same priors as the static model on each param.
    lp += norm.logpdf(theta[0], config.PRIOR_LOG_BPT_MEAN, config.PRIOR_LOG_BPT_SD)
    lp += norm.logpdf(theta[1], config.PRIOR_LOG_SLOP_FRAC_MEAN,
                      config.PRIOR_LOG_SLOP_FRAC_SD)
    lp += norm.logpdf(theta[2], config.PRIOR_LOG_SLOP_SPREAD_MEAN,
                      config.PRIOR_LOG_SLOP_SPREAD_SD)
    lp += hazard.log_prior(theta[3:3 + hazard.n_params])

    # Deltas: Normal(0, 2)
    lp += norm.logpdf(theta[N_STATIC:N_STATIC + N_LEARNING],
                      PRIOR_DELTA_MEAN, PRIOR_DELTA_SD).sum()
    # log tau: Normal(log 30, 1.5)  i.e. tau ~ Lognormal(median=30, sigma=1.5)
    lp += norm.logpdf(theta[N_STATIC + N_LEARNING:N_PARAMS],
                      PRIOR_LOG_TAU_MEAN, PRIOR_LOG_TAU_SD).sum()
    return float(lp)


def log_posterior(theta, data, hazard):
    lp = log_prior(theta, hazard)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, data, hazard)
    return lp + ll


def initial_theta(hazard):
    """Asymptotes at prior means, Delta=0, log tau at prior mean.

    Delta=0 makes the initial model collapse to the static prior-mean fit --
    a sensible starting point that doesn't bake in a learning direction.
    """
    out = np.empty(N_PARAMS)
    out[0] = config.PRIOR_LOG_BPT_MEAN
    out[1] = config.PRIOR_LOG_SLOP_FRAC_MEAN
    out[2] = config.PRIOR_LOG_SLOP_SPREAD_MEAN
    out[3:3 + hazard.n_params] = hazard.initial_theta_slice()
    out[N_STATIC:N_STATIC + N_LEARNING] = PRIOR_DELTA_MEAN
    out[N_STATIC + N_LEARNING:N_PARAMS] = PRIOR_LOG_TAU_MEAN
    return out


def m_clear_at_n(theta, hazard, n):
    """M_clear evaluated using theta(n). Useful for plotting M_clear evolution."""
    return model.m_clear(static_theta_at_n(theta, n), hazard)
