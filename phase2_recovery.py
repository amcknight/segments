"""Phase 2 -- Learning-truth recovery.

Generate data with KNOWN learning curves, fit the learning model, and report
how close MAP gets to truth. Optionally across seeds and at multiple N.

This is the most informative single test we can run right now: it asks the
model to recover something it should be able to recover, on data we control,
with realistic beta-mixture timing structure (the same misspec we'll hit on
real data).

MAP-only -- Hessian / CIs not in scope yet (per the plan). MAP distance from
truth is a *lower bound* on what the posterior knows: wider bands could
comfortably cover truth even when MAP doesn't. But repeated MAP scatter
across seeds at fixed N is a fair stand-in for "what would the typical fit
look like in practice."

Usage:
    python phase2_recovery.py                    # default: 3 seeds x {150,500}
    python phase2_recovery.py --ns 150
    python phase2_recovery.py --seeds 5 --ns 100 200 500
"""
import argparse
import csv
import time

import numpy as np

import learning_model
from generate_synthetic_learning import TRUTH_DEFAULTS, generate, curve
from phase1_check import find_map_learning


PARAM_NAMES = [
    ('bpt (ms)',     None,                None),         # static
    ('sf_inf',       'log_sf_inf',        'sf_inf'),
    ('ssp_inf',      'log_ssp_inf',       'ssp_inf'),
    ('alpha_haz_inf','log_alpha_inf',     'alpha_inf'),
    ('Delta_sf',     'delta_sf',          'sf_delta'),
    ('Delta_ssp',    'delta_ssp',         'ssp_delta'),
    ('Delta_alpha',  'delta_alpha',       'alpha_delta'),
    ('tau_sf',       'log_tau_sf',        'sf_tau'),
    ('tau_ssp',      'log_tau_ssp',       'ssp_tau'),
    ('tau_alpha',    'log_tau_alpha',     'alpha_tau'),
]


def truth_to_theta(truth):
    """Pack the truth dict into a 10-vector matching learning_model.theta layout."""
    return np.array([
        np.log(truth['bpt']),
        np.log(truth['sf_inf']),
        np.log(truth['ssp_inf']),
        np.log(truth['alpha_inf']),
        truth['sf_delta'],
        truth['ssp_delta'],
        truth['alpha_delta'],
        np.log(truth['sf_tau']),
        np.log(truth['ssp_tau']),
        np.log(truth['alpha_tau']),
    ])


def theta_to_natural(theta):
    """Display-units vector: [bpt_ms, sf, ssp, alpha, dsf, dssp, dalpha, tau_sf, tau_ssp, tau_alpha]"""
    return np.array([
        np.exp(theta[0]),
        np.exp(theta[1]),
        np.exp(theta[2]),
        np.exp(theta[3]),
        theta[4],
        theta[5],
        theta[6],
        np.exp(theta[7]),
        np.exp(theta[8]),
        np.exp(theta[9]),
    ])


def m_clear_truth_at_n(truth, hazard, n):
    """Build an effective static-theta from truth at attempt n, then m_clear."""
    log_sf_n = curve(np.log(truth['sf_inf']),
                     truth['sf_delta'], truth['sf_tau'], n)
    log_ssp_n = curve(np.log(truth['ssp_inf']),
                      truth['ssp_delta'], truth['ssp_tau'], n)
    log_alpha_n = curve(np.log(truth['alpha_inf']),
                        truth['alpha_delta'], truth['alpha_tau'], n)
    theta_eff = np.array([np.log(truth['bpt']), log_sf_n, log_ssp_n, log_alpha_n])
    import model
    return model.m_clear(theta_eff, hazard)


def run_one(n_attempts, seed, truth, hazard):
    data = generate(n_attempts, truth, seed=seed)
    n_died = sum(1 for o, _ in data if o == 'died')

    t0 = time.perf_counter()
    theta_map, result = find_map_learning(data, hazard)
    dt = time.perf_counter() - t0
    return {
        'n_attempts': n_attempts,
        'seed': seed,
        'n_died': n_died,
        'theta_map': theta_map,
        'dt': dt,
        'nit': result.nit,
        'converged': bool(result.success),
    }


def summarize(results, truth, hazard):
    theta_truth = truth_to_theta(truth)
    nat_truth = theta_to_natural(theta_truth)

    print(f"\n{'='*78}")
    print(f"Truth (natural units):")
    for i, (label, _, _) in enumerate(PARAM_NAMES):
        print(f"  {label:>14}  {nat_truth[i]:>10.4f}")

    # Group by N
    ns = sorted({r['n_attempts'] for r in results})
    for n in ns:
        rs = [r for r in results if r['n_attempts'] == n]
        print(f"\n--- N={n} ({len(rs)} seeds) ---")
        mats = np.array([theta_to_natural(r['theta_map']) for r in rs])
        mean_map = mats.mean(axis=0)
        std_map = mats.std(axis=0)
        print(f"  {'param':>14}  {'truth':>10}  {'MAP mean':>10}  "
              f"{'MAP std':>10}  {'bias':>10}  {'|bias|/std':>10}")
        for i, (label, _, _) in enumerate(PARAM_NAMES):
            bias = mean_map[i] - nat_truth[i]
            zlike = abs(bias) / std_map[i] if std_map[i] > 1e-9 else float('inf')
            print(f"  {label:>14}  {nat_truth[i]:>10.4f}  {mean_map[i]:>10.4f}  "
                  f"{std_map[i]:>10.4f}  {bias:>+10.4f}  {zlike:>10.2f}")

        # M_clear vs truth at several anchor n
        anchors = sorted({1, n // 4, n // 2, n})
        print(f"\n  M_clear (s):")
        print(f"  {'n':>4}  {'truth':>10}  " + "  ".join(
            f"{'seed' + str(r['seed']):>10}" for r in rs))
        for n_eval in anchors:
            truth_mc = m_clear_truth_at_n(truth, hazard, n_eval) / 1000.0
            row = f"  {n_eval:>4}  {truth_mc:>10.2f}  "
            for r in rs:
                fit_mc = learning_model.m_clear_at_n(
                    r['theta_map'], hazard, n_eval) / 1000.0
                row += f"  {fit_mc:>10.2f}"
            print(row)

        timings = [r['dt'] for r in rs]
        print(f"\n  fit times: " + ", ".join(f"{t:.1f}s" for t in timings))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', type=int, nargs='+', default=[150, 500])
    ap.add_argument('--seeds', type=int, default=3,
                    help='number of seeds per N (uses 0..seeds-1)')
    args = ap.parse_args()

    truth = dict(TRUTH_DEFAULTS)
    hazard = learning_model.haz1()

    print(f"Phase 2 recovery sweep")
    print(f"  truth values: per learning_curves.md spec")
    print(f"  N in {args.ns},  seeds in {list(range(args.seeds))}")
    print(f"  hazard: haz1 (K=1 piecewise);  generator: beta-mixture timing")
    print()

    results = []
    for n in args.ns:
        for seed in range(args.seeds):
            print(f"[N={n}, seed={seed}] generating + fitting...", flush=True)
            r = run_one(n, seed, truth, hazard)
            print(f"  -> {r['nit']} iters in {r['dt']:.1f}s, "
                  f"converged={r['converged']}, {r['n_died']} died")
            results.append(r)

    summarize(results, truth, hazard)


if __name__ == '__main__':
    main()
