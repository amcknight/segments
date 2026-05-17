"""Experiment: how much does data on one segment tighten the prior for
another segment, under various pooling assumptions?

Pipeline:
    1. Generate 500-attempt synth on segment A (truth from spec)
    2. Fit MAP and compute Laplace covariance to get the posterior on
       segment A's latents.
    3. Compute M_clear at n=0 under three scenarios:
         (a) Original prior  (no data, no pooling -- this is v1)
         (b) Segment-A posterior  (data on A, with A)
         (c) Pooled prior for segment B given A's posterior, for various
             sigma_global values modeling cross-segment variation. sigma_global=0
             is full pooling (segments identical); sigma_global=∞ recovers (a).

Output: side-by-side quantile summary plus a tiny histogram per scenario
written to a single PNG so you can eyeball the tightening.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import config
import learning_model
import model
from inference import finite_diff_hessian, _initial_simplex, _bounds
from phase1_check import find_map_learning
from generate_synthetic_learning import generate, TRUTH_DEFAULTS


N_PRIOR = 5000
N_POSTERIOR = 5000
SIGMA_GLOBAL_VALUES = [0.0, 0.25, 0.5, 1.0]  # "small" cross-segment variation


def learning_nll(theta, data, hazard):
    lp = learning_model.log_posterior(theta, data, hazard)
    if not np.isfinite(lp):
        return config.NLL_PENALTY
    return -lp


def learning_laplace_cov(theta_map, data, hazard):
    H = finite_diff_hessian(
        lambda t: learning_nll(t, data, hazard),
        theta_map, eps=config.HESSIAN_FD_EPS,
    )
    if not np.all(np.isfinite(H)):
        return None, False
    eigvals = np.linalg.eigvalsh(H)
    ok = bool(np.min(eigvals) > 0)
    if not ok:
        # Heavy regularization: ensure all eigenvalues >= 0.5 so marginal
        # variance is at most ~2. Otherwise the flat (Delta, tau) ridge
        # gives infinite-variance directions and the Laplace samples are
        # mostly pathological. This regularized cov is BIASED -- bands
        # are too narrow on the flat directions -- but bounded.
        shift = max(0.0, 0.5 - float(np.min(eigvals)))
        H = H + shift * np.eye(len(theta_map))
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
        ok = False
    return cov, ok


def run_metropolis(start, log_post_fn, n_iters, proposal_sd, rng):
    """Random-walk Metropolis-Hastings. Returns (chain, accept_rate)."""
    d = len(start)
    chain = np.zeros((n_iters, d))
    chain[0] = start
    lp = log_post_fn(start)
    accepted = 0
    for i in range(1, n_iters):
        prop = chain[i-1] + rng.normal(0, proposal_sd, d)
        lp_new = log_post_fn(prop)
        if np.isfinite(lp_new) and (lp_new >= lp
                                     or np.log(rng.random()) < lp_new - lp):
            chain[i] = prop
            lp = lp_new
            accepted += 1
        else:
            chain[i] = chain[i-1]
    return chain, accepted / n_iters


def m_clear_at_n_from_theta(theta, n, hazard):
    """Returns seconds. For thetas where p_surv underflows to 0
    (alpha_haz huge), returns +inf -- we'd literally never clear."""
    try:
        ms = learning_model.m_clear_at_n(theta, hazard, n)
        if not np.isfinite(ms) or ms < 0:
            return np.inf
        return ms / 1000.0
    except (ZeroDivisionError, FloatingPointError, OverflowError):
        return np.inf


def sample_prior_thetas(n_samples, rng):
    """Same priors as the inspector: independent Normals on each latent."""
    mu = np.array([
        config.PRIOR_LOG_BPT_MEAN,
        config.PRIOR_LOG_SLOP_FRAC_MEAN,
        config.PRIOR_LOG_SLOP_SPREAD_MEAN,
        config.PRIOR_LOG_HAZ_MEAN,
        learning_model.PRIOR_DELTA_MEAN,
        learning_model.PRIOR_DELTA_MEAN,
        learning_model.PRIOR_DELTA_MEAN,
        learning_model.PRIOR_LOG_TAU_MEAN,
        learning_model.PRIOR_LOG_TAU_MEAN,
        learning_model.PRIOR_LOG_TAU_MEAN,
    ])
    sd = np.array([
        config.PRIOR_LOG_BPT_SD,
        config.PRIOR_LOG_SLOP_FRAC_SD,
        config.PRIOR_LOG_SLOP_SPREAD_SD,
        config.PRIOR_LOG_HAZ_SD,
        learning_model.PRIOR_DELTA_SD,
        learning_model.PRIOR_DELTA_SD,
        learning_model.PRIOR_DELTA_SD,
        learning_model.PRIOR_LOG_TAU_SD,
        learning_model.PRIOR_LOG_TAU_SD,
        learning_model.PRIOR_LOG_TAU_SD,
    ])
    return rng.normal(mu, sd, size=(n_samples, 10))


def m_clear_array(thetas, n, hazard):
    return np.array([m_clear_at_n_from_theta(t, n, hazard) for t in thetas])


def fmt_seconds(s):
    if not np.isfinite(s):
        return "inf"
    if s >= 1e15:
        return f"{s:.1e} s"
    if s >= 365 * 24 * 3600:
        return f"{s / (365 * 24 * 3600):.1g} yr"
    if s >= 24 * 3600:
        return f"{s / (24 * 3600):.1f} d"
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def quantile_summary(vals, label):
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    vals = np.asarray(vals, dtype=float)
    # Treat inf as a very large finite number for quantile computation,
    # but report fraction-infinite separately.
    n_inf = int(np.sum(~np.isfinite(vals)))
    pct_inf = 100.0 * n_inf / len(vals)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        qvals = [np.inf] * 5
    else:
        # Use the finite-only data; then re-mix back the inf fraction.
        sorted_v = np.sort(vals)  # nans/infs go to the end
        qvals = [sorted_v[min(int(q * len(sorted_v)), len(sorted_v) - 1)]
                 for q in qs]
    inf_tag = f"  ({pct_inf:.1f}% impossible)" if pct_inf > 0.5 else ""
    return (
        f"{label:>42}  "
        f"5%={fmt_seconds(qvals[0]):>10}  "
        f"25%={fmt_seconds(qvals[1]):>10}  "
        f"50%={fmt_seconds(qvals[2]):>10}  "
        f"75%={fmt_seconds(qvals[3]):>10}  "
        f"95%={fmt_seconds(qvals[4]):>12}{inf_tag}"
    )


def main():
    rng = np.random.default_rng(0)
    hazard = learning_model.haz1()
    n_target = 0   # the attempt number we care about for "first try"

    # --- (1) Original prior (no data, no pooling) -----------------------------
    print("Sampling original prior...")
    prior_thetas = sample_prior_thetas(N_PRIOR, rng)
    prior_m_clear = m_clear_array(prior_thetas, n_target, hazard)

    # --- (2) Fit segment A and get Laplace posterior --------------------------
    print(f"Generating segment A (N=500, seed=1) and fitting...")
    data_a = generate(500, TRUTH_DEFAULTS, seed=1)
    t0 = time.perf_counter()
    theta_map_a, result_a = find_map_learning(data_a, hazard)
    dt_fit = time.perf_counter() - t0
    print(f"  MAP in {dt_fit:.1f}s, {result_a.nit} iters, success={result_a.success}")

    print("  Computing regularized Laplace covariance for MH proposal SDs...")
    t0 = time.perf_counter()
    cov_a, pd = learning_laplace_cov(theta_map_a, data_a, hazard)
    dt_hess = time.perf_counter() - t0
    print(f"  Hessian in {dt_hess:.1f}s, positive-definite without reg={pd}")

    # Tight uniform proposal: at N=500 the posterior is sharply peaked.
    proposal_sd = np.full(10, 0.015)
    print(f"  Proposal SDs (uniform): {proposal_sd[0]}")

    n_iters = 6000
    print(f"  Running Metropolis-Hastings ({n_iters} iters)...")
    t0 = time.perf_counter()
    chain, acc = run_metropolis(
        theta_map_a,
        lambda t: learning_model.log_posterior(t, data_a, hazard),
        n_iters=n_iters,
        proposal_sd=proposal_sd,
        rng=rng,
    )
    dt_mh = time.perf_counter() - t0
    print(f"  MH in {dt_mh:.1f}s, acceptance rate={acc:.2%}")
    # Burn-in first half, then use the rest
    posterior_a_thetas = chain[n_iters // 2:]
    print(f"  Using {len(posterior_a_thetas)} posterior samples (post-burnin, thinned)")
    posterior_a_m_clear = m_clear_array(posterior_a_thetas, n_target, hazard)

    # --- (3) Pooled prior on segment B for various sigma_global -------------------
    pooled_results = {}
    N_post = len(posterior_a_thetas)
    for sg in SIGMA_GLOBAL_VALUES:
        # Each segment B's theta = posterior A's theta + Normal(0, sigma_global) per coord
        jitter = rng.normal(0, sg, size=(N_post, 10))
        seg_b_thetas = posterior_a_thetas + jitter
        pooled_results[sg] = m_clear_array(seg_b_thetas, n_target, hazard)

    # --- Print summary --------------------------------------------------------
    print()
    print(f"M_clear at n={n_target} (i.e., first-attempt expected wall-clock to clear):")
    print()
    print(quantile_summary(prior_m_clear,
        "Original prior (v1, no pooling)"))
    print(quantile_summary(posterior_a_m_clear,
        "Segment A posterior (after 500 attempts)"))
    for sg in SIGMA_GLOBAL_VALUES:
        if sg == 0.0:
            tag = f"Seg B prior, sigma_global=0 (full pooling)"
        else:
            tag = f"Seg B prior, sigma_global={sg}"
        print(quantile_summary(pooled_results[sg], tag))

    truth_theta = np.array([
        np.log(TRUTH_DEFAULTS['bpt']),
        np.log(TRUTH_DEFAULTS['sf_inf']),
        np.log(TRUTH_DEFAULTS['ssp_inf']),
        np.log(TRUTH_DEFAULTS['alpha_inf']),
        TRUTH_DEFAULTS['sf_delta'], TRUTH_DEFAULTS['ssp_delta'],
        TRUTH_DEFAULTS['alpha_delta'],
        np.log(TRUTH_DEFAULTS['sf_tau']), np.log(TRUTH_DEFAULTS['ssp_tau']),
        np.log(TRUTH_DEFAULTS['alpha_tau']),
    ])
    truth_m_clear = m_clear_at_n_from_theta(truth_theta, n_target, hazard)
    print()
    print(f"Truth (segment A): M_clear(n={n_target}) = "
          f"{truth_m_clear:.1f} s  ({fmt_seconds(truth_m_clear)})")

    # --- Plot histograms ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plot_specs = [
        ('Original prior\n(v1: no data, no pooling)', prior_m_clear),
        ('Segment A posterior\n(data on A)', posterior_a_m_clear),
        (f'Seg B prior under FULL pooling\n(sigma_global=0)', pooled_results[0.0]),
        (f'Seg B prior, sigma_global=0.25', pooled_results[0.25]),
        (f'Seg B prior, sigma_global=0.5', pooled_results[0.5]),
        (f'Seg B prior, sigma_global=1.0\n(~back to original prior)', pooled_results[1.0]),
    ]
    for ax, (title, vals) in zip(axes.flat, plot_specs):
        vals = np.asarray(vals)
        clipped = vals[np.isfinite(vals) & (vals > 0) & (vals < 1e10)]
        ax.hist(np.log10(clipped), bins=50, color='gray', alpha=0.7)
        ax.axvline(np.log10(max(truth_m_clear, 1e-6)),
                   color='#e67e22', lw=2, label=f'truth ({fmt_seconds(truth_m_clear)})')
        qmed = np.median(vals)
        ax.axvline(np.log10(max(qmed, 1e-6)),
                   color='#2980b9', lw=1.5, ls='--',
                   label=f'median ({fmt_seconds(qmed)})')
        ax.set_xlabel(r'$\log_{10}$ M_clear (s)')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    fig.suptitle('M_clear at n=0 (first-try expected wall-clock to clear): '
                 'how data on segment A tightens the implied prior on segment B',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig('pooling_experiment.png', dpi=110, bbox_inches='tight')
    print(f"\nWrote histograms to pooling_experiment.png")


if __name__ == '__main__':
    main()
