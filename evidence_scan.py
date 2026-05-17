"""Compute Laplace log-evidence at multiple K values per attempt and apply the
evidence-threshold promotion rule.

For each attempt n and each K in KS:
    1. Fit MAP (warm-started from the previous attempt's MAP at the same K).
    2. Compute Laplace covariance and log-evidence.
The promotion rule then walks forward in attempts and ratchets K upward when
log_evidence(K+1) - log_evidence(K) > delta.

Usage:
    python evidence_scan.py
    python evidence_scan.py --delta 1.0 --kmax 3
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

import config
import binning
from hazard import PiecewiseConstantHazard
from inference import find_map, laplace_cov, log_evidence_laplace


DATA_PATH = 'test_data.tsv'
DEFAULT_DELTA = 1.0
DEFAULT_KMAX = 3
FIG_WIDTH = 12
FIG_HEIGHT = 10
FIG_DPI = 110


def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append((r['outcome'].strip(), float(r['time_ms'])))
    return rows


def evidence_scan(data, Ks):
    """For each K in Ks, fit MAP + Laplace at each prefix of the data.
    Returns dict K -> array of log-evidences of length n_attempts."""
    evidences = {K: [] for K in Ks}
    theta_prevs = {K: None for K in Ks}
    seen = []

    header = f"{'#':>3} {'outcome':>9} {'time':>6}"
    for K in Ks:
        header += f"  {'evK=' + str(K):>12}"
    print(header)

    for i, (outcome, time_ms) in enumerate(data, 1):
        seen.append((outcome, time_ms))
        row = f"{i:>3} {outcome:>9} {time_ms:>6.0f}"
        for K in Ks:
            hazard = PiecewiseConstantHazard(K, binning.uniform(K))
            theta_map, _ = find_map(seen, hazard, theta_init=theta_prevs[K])
            theta_prevs[K] = theta_map
            cov, _ = laplace_cov(theta_map, seen, hazard)
            ev = log_evidence_laplace(theta_map, cov, seen, hazard)
            evidences[K].append(ev)
            row += f"  {ev:>12.2f}"
        print(row)
    return {K: np.array(v) for K, v in evidences.items()}


def apply_promotion_rule(evidences, Ks, delta):
    """One-way ratchet: K starts at min(Ks); at each attempt, while
    evidence(K+1) - evidence(K) > delta and K+1 is available, promote."""
    n = len(evidences[Ks[0]])
    K_curr = min(Ks)
    K_history = []
    for i in range(n):
        while K_curr < max(Ks):
            gain = evidences[K_curr + 1][i] - evidences[K_curr][i]
            if np.isfinite(gain) and gain > delta:
                K_curr += 1
            else:
                break
        K_history.append(K_curr)
    return np.array(K_history)


def plot_results(evidences, Ks, K_history, delta, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
    n = len(K_history)
    xs = np.arange(1, n + 1)

    for K in Ks:
        axes[0].plot(xs, evidences[K], lw=1.5, marker='.', label=f'K={K}')
    axes[0].set_ylabel('log evidence (Laplace)')
    axes[0].set_title('Per-attempt log marginal likelihood at each K')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(alpha=0.3)

    for k in range(len(Ks) - 1):
        diff = evidences[Ks[k + 1]] - evidences[Ks[k]]
        axes[1].plot(xs, diff, lw=1.5, marker='.',
                     label=f'K={Ks[k+1]} − K={Ks[k]}')
    axes[1].axhline(delta, color='k', ls='--', lw=1, alpha=0.6,
                    label=f'δ = {delta}')
    axes[1].axhline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
    axes[1].set_ylabel('Δ log evidence (gain from extra bin)')
    axes[1].set_title('Evidence gain from adding one more bin')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].step(xs, K_history, where='post', lw=2.0, color='C2')
    axes[2].set_xlabel('attempt #')
    axes[2].set_ylabel('K (selected)')
    axes[2].set_yticks(Ks)
    axes[2].set_title(f'K under threshold rule (δ={delta}, one-way ratchet)')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI)
    print(f"\nSaved figure to {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', type=float, default=DEFAULT_DELTA)
    ap.add_argument('--kmax', type=int, default=DEFAULT_KMAX)
    ap.add_argument('--save', default='pic/evidence_scan.png')
    args = ap.parse_args()

    Ks = list(range(1, args.kmax + 1))
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} attempts. Scanning K in {Ks}, delta={args.delta}.\n")

    evidences = evidence_scan(data, Ks)
    K_history = apply_promotion_rule(evidences, Ks, args.delta)

    print()
    print(f"K trajectory under delta={args.delta}: {list(K_history)}")
    promotions = [(i + 1, K_history[i]) for i in range(1, len(K_history))
                  if K_history[i] != K_history[i - 1]]
    if promotions:
        for attempt_i, K_new in promotions:
            print(f"  promoted to K={K_new} at attempt {attempt_i}")
    else:
        print(f"  no promotions; stayed at K={K_history[0]}")

    plot_results(evidences, Ks, K_history, args.delta, args.save)


if __name__ == '__main__':
    main()
