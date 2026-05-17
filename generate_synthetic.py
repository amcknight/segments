"""Generate synthetic attempts from the speedrun segment model.

Used to test fit quality at scales bigger than the current real dataset
(~22 attempts) while waiting for more real data. The generative process
mirrors segment_model.md:

    T_clean = bpt + Lognormal(mu_log, sigma_log)
    if Bernoulli(alpha):
        component k ~ Categorical(w)
        S_death ~ Beta(mu_k * kappa_k, (1 - mu_k) * kappa_k)
        time_obs = S_death * T_clean + respawn      (outcome='died')
    else:
        time_obs = T_clean                          (outcome='survived')

The defaults pick a two-mode death distribution that loosely matches the
existing test_data.tsv (one strong central cluster, one smaller late cluster)
so the synthetic feels comparable. Override on the CLI if you want different
ground truth.

Usage:
    python generate_synthetic.py --n 150 --out test_synth.tsv
    python generate_synthetic.py --n 300 --seed 42 --alpha 0.6 \
        --mu 0.3 0.7 --kappa 20 10 --w 0.65 0.35
"""
import argparse
import csv

import numpy as np

import config


DEFAULTS = dict(
    bpt=23000.0,
    slop_frac=0.25,
    slop_spread=0.30,
    alpha=0.65,
    mu=[0.35, 0.70],
    kappa=[15.0, 10.0],
    w=[0.60, 0.40],
    respawn_ms=config.RESPAWN_MS,
)


def lognormal_slop_params(slop_frac, slop_spread, bpt):
    """Match moments: mean(slop)=slop_frac*bpt, std(slop)=slop_spread*mean(slop)."""
    mean_slop = slop_frac * bpt
    sigma_log_sq = np.log(1.0 + slop_spread ** 2)
    mu_log = np.log(mean_slop) - 0.5 * sigma_log_sq
    return mu_log, np.sqrt(sigma_log_sq)


def generate(n, bpt, slop_frac, slop_spread, alpha, mu, kappa, w,
             respawn_ms, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    w = np.asarray(w, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w} (sum {w.sum():.4f})")
    if not (len(mu) == len(kappa) == len(w)):
        raise ValueError(f"mu/kappa/w must have same length, "
                         f"got {len(mu)}/{len(kappa)}/{len(w)}")

    mu_log, sigma_log = lognormal_slop_params(slop_frac, slop_spread, bpt)

    rows = []
    K = len(mu)
    a = mu * kappa
    b = (1.0 - mu) * kappa
    for _ in range(n):
        t_clean = bpt + rng.lognormal(mu_log, sigma_log)
        if rng.random() < alpha:
            k = rng.choice(K, p=w)
            s_death = rng.beta(a[k], b[k])
            t_obs = s_death * t_clean + respawn_ms
            rows.append(('died', round(float(t_obs))))
        else:
            rows.append(('survived', round(float(t_clean))))
    return rows


def write_tsv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['outcome', 'time_ms'])
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bpt', type=float, default=DEFAULTS['bpt'])
    ap.add_argument('--slop_frac', type=float, default=DEFAULTS['slop_frac'])
    ap.add_argument('--slop_spread', type=float, default=DEFAULTS['slop_spread'])
    ap.add_argument('--alpha', type=float, default=DEFAULTS['alpha'])
    ap.add_argument('--mu', type=float, nargs='+', default=DEFAULTS['mu'])
    ap.add_argument('--kappa', type=float, nargs='+', default=DEFAULTS['kappa'])
    ap.add_argument('--w', type=float, nargs='+', default=DEFAULTS['w'])
    ap.add_argument('--out', default='test_synth.tsv')
    args = ap.parse_args()

    rows = generate(args.n, args.bpt, args.slop_frac, args.slop_spread,
                    args.alpha, args.mu, args.kappa, args.w,
                    DEFAULTS['respawn_ms'], seed=args.seed)
    write_tsv(rows, args.out)
    n_died = sum(1 for o, _ in rows if o == 'died')
    print(f"Wrote {len(rows)} attempts ({n_died} died, "
          f"{len(rows) - n_died} survived) to {args.out}")
    print("\nTrue parameters:")
    print(f"  bpt = {args.bpt:.0f}ms,  slop_frac = {args.slop_frac:.3f},  "
          f"slop_spread = {args.slop_spread:.3f}")
    print(f"  alpha = {args.alpha:.3f}  ({args.alpha*100:.0f}% die)")
    for i, (m, k, w) in enumerate(zip(args.mu, args.kappa, args.w), 1):
        print(f"  comp {i}:  mu = {m:.3f},  kappa = {k:.2f},  w = {w:.3f}")


if __name__ == '__main__':
    main()
