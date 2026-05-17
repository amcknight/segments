"""Visual preview of a synthetic death-density ground truth.

Plots the per-component decomposition (like beta_compare's bottom-left panel)
PLUS the implied hazard, survival, and a sample histogram for the chosen
generative parameters. Use this to iterate on a synthetic ground truth before
calling generate_synthetic.py to actually create the .tsv.

Usage:
    python preview_synthetic.py \\
        --mu 0.3 0.55 0.8  --kappa 5 60 8  --w 0.4 0.2 0.4  --alpha 0.65
    python preview_synthetic.py --n 300 --seed 1   # also draw a sample hist

The script does not fit anything. It just draws the truth.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.special import betainc

import config


def f_components(s, mu, kappa, w, alpha):
    """alpha * w_k * Beta_pdf(s | a_k, b_k) for each k, shape (K, len(s))."""
    a = mu * kappa
    b = (1.0 - mu) * kappa
    comp = beta_dist.pdf(s[:, None], a[None, :], b[None, :])  # (n_s, K)
    return alpha * (w[None, :] * comp).T                       # (K, n_s)


def f_total(s, mu, kappa, w, alpha):
    return f_components(s, mu, kappa, w, alpha).sum(axis=0)


def F_total(s, mu, kappa, w, alpha):
    a = mu * kappa
    b = (1.0 - mu) * kappa
    comp = betainc(a[None, :], b[None, :], s[:, None])
    return alpha * (w[None, :] * comp).sum(axis=1)


def hazard(s, mu, kappa, w, alpha):
    return f_total(s, mu, kappa, w, alpha) / (1.0 - F_total(s, mu, kappa, w, alpha))


def sample(n, mu, kappa, w, alpha, bpt, slop_frac, slop_spread, respawn_ms,
           seed=0):
    """Draw n synthetic attempts under the speedrun generative model."""
    rng = np.random.default_rng(seed)
    sigma_log_sq = np.log(1.0 + slop_spread ** 2)
    mu_log = np.log(slop_frac * bpt) - 0.5 * sigma_log_sq
    sigma_log = np.sqrt(sigma_log_sq)
    a = mu * kappa
    b = (1.0 - mu) * kappa
    K = len(mu)

    s_star = []
    outcomes = []
    times = []
    for _ in range(n):
        t_clean = bpt + rng.lognormal(mu_log, sigma_log)
        if rng.random() < alpha:
            k = rng.choice(K, p=w)
            s_death = rng.beta(a[k], b[k])
            outcomes.append('died')
            times.append(s_death * t_clean + respawn_ms)
            s_star.append(s_death)
        else:
            outcomes.append('survived')
            times.append(t_clean)
    return np.array(s_star), outcomes, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, nargs='+',
                    default=[0.3, 0.55, 0.8],
                    help='component means (length K)')
    ap.add_argument('--kappa', type=float, nargs='+',
                    default=[5.0, 60.0, 8.0],
                    help='component concentrations (length K)')
    ap.add_argument('--w', type=float, nargs='+',
                    default=[0.4, 0.2, 0.4],
                    help='component weights (length K, must sum to 1)')
    ap.add_argument('--alpha', type=float, default=0.65,
                    help='P(die in an attempt)')
    ap.add_argument('--bpt', type=float, default=23000.0)
    ap.add_argument('--slop_frac', type=float, default=0.25)
    ap.add_argument('--slop_spread', type=float, default=0.30)
    ap.add_argument('--n', type=int, default=0,
                    help='if > 0, also draw N synthetic attempts and overlay '
                         's* histogram')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default='pic/preview_synth.png')
    args = ap.parse_args()

    mu = np.array(args.mu, dtype=float)
    kappa = np.array(args.kappa, dtype=float)
    w = np.array(args.w, dtype=float)
    K = len(mu)
    if not (len(kappa) == len(w) == K):
        raise ValueError(f"mu/kappa/w must all have same length, "
                         f"got {len(mu)}/{len(kappa)}/{len(w)}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got w={w} sum={w.sum():.4f}")

    s_grid = np.linspace(1e-4, 1 - 1e-4, 400)
    comps = f_components(s_grid, mu, kappa, w, args.alpha)
    total = comps.sum(axis=0)
    surv = 1.0 - F_total(s_grid, mu, kappa, w, args.alpha)
    haz = hazard(s_grid, mu, kappa, w, args.alpha)

    # diagnostic numbers
    p_surv = 1.0 - args.alpha
    mean_clean = args.bpt + args.slop_frac * args.bpt
    e_s_given_died = float((s_grid * total).sum() * (s_grid[1] - s_grid[0])
                            / args.alpha)
    e_died_time = e_s_given_died * mean_clean + config.RESPAWN_MS
    m_clear_s = ((1.0 - p_surv) / p_surv * e_died_time + mean_clean) / 1000.0

    print(f"K={K} components, alpha={args.alpha}, sum(w)={w.sum():.3f}")
    for k in range(K):
        a_k, b_k = mu[k] * kappa[k], (1 - mu[k]) * kappa[k]
        spike = a_k < 1.0 or b_k < 1.0
        spike_warn = '  [SPIKE: Beta(a<1 or b<1) diverges at boundary]' if spike else ''
        print(f"  comp {k+1}: mu={mu[k]:.3f}  kappa={kappa[k]:.2f}  "
              f"w={w[k]:.3f}  (a={a_k:.2f}, b={b_k:.2f}){spike_warn}")
    print(f"\nDerived under (bpt={args.bpt:.0f}, "
          f"slop_frac={args.slop_frac}, respawn={config.RESPAWN_MS/1000:.1f}s):")
    print(f"  P(die per attempt) = {args.alpha:.3f}")
    print(f"  E[s_death | died]  = {e_s_given_died:.3f}")
    print(f"  mean_clean         = {mean_clean/1000:.2f}s")
    print(f"  M_clear (expected total time to first clear) = {m_clear_s:.2f}s")

    # optional sample
    s_stars = None
    if args.n > 0:
        s_stars, outcomes, times = sample(
            args.n, mu, kappa, w, args.alpha,
            args.bpt, args.slop_frac, args.slop_spread,
            config.RESPAWN_MS, seed=args.seed,
        )
        n_died = sum(1 for o in outcomes if o == 'died')
        print(f"\nSampled {args.n} attempts: {n_died} died "
              f"({100*n_died/args.n:.0f}%), {args.n-n_died} survived")

    # ------ figure ------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_dec, ax_haz, ax_S, ax_total = axes.flat

    # decomposition
    for k in range(K):
        a_k, b_k = mu[k] * kappa[k], (1 - mu[k]) * kappa[k]
        ax_dec.plot(s_grid, comps[k], lw=1.5, color=f'C{k+2}',
                    label=f'comp {k+1}:  mu={mu[k]:.2f}, kap={kappa[k]:.1f}, '
                          f'w={w[k]:.2f}')
    ax_dec.plot(s_grid, total, lw=2.5, color='k',
                label=f'total (alpha={args.alpha:.2f})')
    if s_stars is not None and len(s_stars) > 0:
        ax_dec.hist(s_stars, bins=20, range=(0, 1), density=True,
                    color='gray', alpha=0.25,
                    label=f'sampled s* (n={len(s_stars)})')
    ax_dec.set_xlabel('s')
    ax_dec.set_ylabel('density (weighted)')
    ax_dec.set_title('Decomposition of the synthetic truth')
    ax_dec.legend(fontsize=8)
    ax_dec.grid(alpha=0.3)

    # hazard
    ax_haz.plot(s_grid, haz, lw=2.0, color='C3')
    ax_haz.set_xlabel('s')
    ax_haz.set_ylabel('hazard h(s)')
    ax_haz.set_title('Implied hazard rate')
    ax_haz.grid(alpha=0.3)

    # survival
    ax_S.plot(s_grid, surv, lw=2.0, color='C2')
    ax_S.set_xlabel('s')
    ax_S.set_ylabel('S(s)')
    ax_S.set_title(f'Implied survival (S(1) = 1 - alpha = {1-args.alpha:.2f})')
    ax_S.set_ylim(0, 1.02)
    ax_S.grid(alpha=0.3)

    # total density + cdf
    cdf = F_total(s_grid, mu, kappa, w, args.alpha)
    ax_total.plot(s_grid, total, lw=2.0, color='k', label='f(s) total')
    ax_total.plot(s_grid, cdf, lw=1.5, color='C0', ls='--',
                  label='F(s) cumulative')
    ax_total.axhline(args.alpha, color='C0', ls=':', lw=1, alpha=0.6,
                     label=f'alpha = {args.alpha:.2f}')
    ax_total.set_xlabel('s')
    ax_total.set_ylabel('density / probability')
    ax_total.set_title('Total density f(s) and CDF F(s)')
    ax_total.legend(fontsize=8)
    ax_total.grid(alpha=0.3)

    fig.suptitle(f'Synthetic ground-truth preview;  '
                 f'M_clear = {m_clear_s:.1f}s')
    fig.tight_layout()
    fig.savefig(args.save, dpi=110)
    print(f"\nSaved preview to {args.save}")


if __name__ == '__main__':
    main()
