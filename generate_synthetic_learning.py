"""Synthetic attempts with per-attempt learning curves.

Mirrors generate_synthetic.py (beta-mixture hazard for within-attempt death
timing, kept STATIC because mu_k/kappa_k don't learn per the spec) but lets
slop_frac, slop_spread, and the haz1 cumulative-hazard 'alpha' evolve over n:

    log theta(n) = log theta_inf + Delta * exp(-n / tau)

Important misspecification note: we generate death TIMING from a beta mixture
(realistic, what we expect on real data) and the model is fit with haz1
(single-rate constant hazard). The Phase 2 recovery test must work despite
this misspec, since that's the regime we'll be in on real data until beta2 is
re-promoted (per learning_curves.md).

The truth values used here come from external_docs/learning_curves.md Phase 2:

    | param        | theta_inf | Delta | tau |
    | slop_frac    | 0.05      | 1.4   | 40  |
    | slop_spread  | 0.2       | 1.0   | 80  |
    | alpha (haz1) | 0.3       | 1.4   | 20  |
    | bpt          | 30000ms (static)         |

Sidecar JSON of truth params is written alongside the .tsv so phase2_recovery
can read it back without any handshake.

Usage:
    python generate_synthetic_learning.py --n 150 --seed 0
    python generate_synthetic_learning.py --n 500 --seed 1 --out custom.tsv
"""
import argparse
import csv
import json
import os

import numpy as np

import config


TRUTH_DEFAULTS = dict(
    bpt=30000.0,
    sf_inf=0.05,
    sf_delta=1.4,
    sf_tau=40.0,
    ssp_inf=0.2,
    ssp_delta=1.0,
    ssp_tau=80.0,
    alpha_inf=0.3,        # haz1 cumulative-hazard "alpha" at asymptote
    alpha_delta=1.4,
    alpha_tau=20.0,
    # Beta-mixture for within-attempt death timing (static, not learning).
    mu=[0.35, 0.70],
    kappa=[15.0, 10.0],
    w=[0.60, 0.40],
    respawn_ms=config.RESPAWN_MS,
)


def curve(log_theta_inf, delta, tau, n):
    """log theta(n) = log theta_inf + delta * exp(-n / tau). Vectorized over n."""
    n = np.asarray(n, dtype=float)
    return log_theta_inf + delta * np.exp(-n / tau)


def lognormal_slop_params(slop_frac, slop_spread, bpt):
    mean_slop = slop_frac * bpt
    sigma_log_sq = np.log(1.0 + slop_spread ** 2)
    mu_log = np.log(mean_slop) - 0.5 * sigma_log_sq
    return mu_log, np.sqrt(sigma_log_sq)


def generate(n_attempts, truth, seed=0):
    rng = np.random.default_rng(seed)

    log_sf_inf = np.log(truth['sf_inf'])
    log_ssp_inf = np.log(truth['ssp_inf'])
    log_alpha_inf = np.log(truth['alpha_inf'])

    mu = np.asarray(truth['mu'], dtype=float)
    kappa = np.asarray(truth['kappa'], dtype=float)
    w = np.asarray(truth['w'], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w}")

    a = mu * kappa
    b = (1.0 - mu) * kappa

    bpt = truth['bpt']
    rows = []
    for n in range(1, n_attempts + 1):
        sf_n = float(np.exp(curve(log_sf_inf, truth['sf_delta'],
                                  truth['sf_tau'], n)))
        ssp_n = float(np.exp(curve(log_ssp_inf, truth['ssp_delta'],
                                   truth['ssp_tau'], n)))
        alpha_haz_n = float(np.exp(curve(log_alpha_inf, truth['alpha_delta'],
                                         truth['alpha_tau'], n)))
        # haz1 cum hazard -> P(die per attempt)
        p_die_n = 1.0 - np.exp(-alpha_haz_n)

        mu_log, sigma_log = lognormal_slop_params(sf_n, ssp_n, bpt)
        t_clean = bpt + rng.lognormal(mu_log, sigma_log)
        if rng.random() < p_die_n:
            k = rng.choice(len(mu), p=w)
            s_death = rng.beta(a[k], b[k])
            t_obs = s_death * t_clean + truth['respawn_ms']
            rows.append(('died', round(float(t_obs))))
        else:
            rows.append(('survived', round(float(t_clean))))
    return rows


def write_outputs(rows, truth, n_attempts, seed, out_path):
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['outcome', 'time_ms'])
        writer.writerows(rows)

    truth_path = os.path.splitext(out_path)[0] + '.truth.json'
    truth_dump = dict(truth)
    truth_dump['n_attempts'] = n_attempts
    truth_dump['seed'] = seed
    with open(truth_path, 'w') as f:
        json.dump(truth_dump, f, indent=2)
    return truth_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='test_synth_learning.tsv')
    # Allow CLI overrides on each truth latent (rarely needed).
    for key, default in TRUTH_DEFAULTS.items():
        if isinstance(default, list):
            ap.add_argument(f'--{key}', type=float, nargs='+', default=default)
        else:
            ap.add_argument(f'--{key}', type=float, default=default)
    args = ap.parse_args()

    truth = {k: getattr(args, k) for k in TRUTH_DEFAULTS}
    rows = generate(args.n, truth, seed=args.seed)
    truth_path = write_outputs(rows, truth, args.n, args.seed, args.out)

    n_died = sum(1 for o, _ in rows if o == 'died')
    print(f"Wrote {len(rows)} attempts ({n_died} died, "
          f"{len(rows) - n_died} survived) -> {args.out}")
    print(f"Truth -> {truth_path}\n")

    # Quick sanity print: per-attempt curves at n=1, N/4, N/2, N (illustrative).
    ns = sorted({1, max(1, args.n // 4), max(1, args.n // 2), args.n})
    print(f"  {'n':>4}  {'sf(n)':>8}  {'ssp(n)':>8}  {'alpha_haz':>10}  "
          f"{'P(die)':>8}")
    for n in ns:
        sf = float(np.exp(curve(np.log(truth['sf_inf']),
                                truth['sf_delta'], truth['sf_tau'], n)))
        ssp = float(np.exp(curve(np.log(truth['ssp_inf']),
                                 truth['ssp_delta'], truth['ssp_tau'], n)))
        ah = float(np.exp(curve(np.log(truth['alpha_inf']),
                                truth['alpha_delta'], truth['alpha_tau'], n)))
        p_die = 1.0 - np.exp(-ah)
        print(f"  {n:>4}  {sf:>8.4f}  {ssp:>8.4f}  {ah:>10.4f}  {p_die:>8.4f}")


if __name__ == '__main__':
    main()
