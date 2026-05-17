"""Compare bin-edge picker strategies at multiple K values on the final dataset.

The Laplace log-evidence is the score. We compute it for each
(picker, K) combination and report the best.

Pickers tried:
    - uniform: equal-width bins
    - quantile: edges at quantiles of s* (known to split clusters; baseline only)
    - kde_valleys: edges at local minima of a Gaussian KDE on s* (the
      'place edges between clusters' strategy)
    - kmeans_midpoints: edges at midpoints between 1D k-means cluster centers

Usage:
    python bin_search.py
"""
import csv

import numpy as np
import matplotlib.pyplot as plt

import config
import binning
from hazard import PiecewiseConstantHazard
from model import unpack, lognormal_params, slop_moments
from inference import find_map, laplace_cov, log_evidence_laplace


DATA_PATH = 'test_data.tsv'
K_VALUES = [1, 2, 3, 4, 5]
KDE_BANDWIDTHS = [0.03, 0.05, 0.08, 0.12]
SAVE_PATH = 'pic/bin_search.png'


def load_data(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return [(r['outcome'].strip(), float(r['time_ms'])) for r in reader]


def fit_evidence(data, K, bin_edges):
    hazard = PiecewiseConstantHazard(K, bin_edges)
    theta_map, _ = find_map(data, hazard)
    cov, ok = laplace_cov(theta_map, data, hazard)
    ev = log_evidence_laplace(theta_map, cov, data, hazard)
    return theta_map, ev, ok


def s_star_values(data, theta_map, K):
    hazard = PiecewiseConstantHazard(K, binning.uniform(K))
    bpt, sf, ssp, _ = unpack(theta_map, hazard)
    mu_log, sigma_log = lognormal_params(sf, ssp, bpt)
    mean_slop, _ = slop_moments(mu_log, sigma_log)
    mean_clean = bpt + mean_slop
    return np.array([
        (t_obs - config.RESPAWN_MS) / mean_clean
        for outcome, t_obs in data if outcome == 'died'
    ])


def main():
    data = load_data(DATA_PATH)
    n_died = sum(1 for o, _ in data if o == 'died')
    print(f"Loaded {len(data)} attempts ({n_died} died).\n")

    print("Fitting K=1 to get mean_clean for s* computation...")
    theta1, ev1, _ = fit_evidence(data, 1, binning.uniform(1))
    s_stars = s_star_values(data, theta1, 1)
    print(f"  s* sorted: {np.sort(s_stars).round(3).tolist()}\n")

    fmt = f"{'picker':>20}  {'K':>2}  {'edges':>55}  {'log evidence':>14}"
    print(fmt)
    print("-" * len(fmt))
    results = []

    # Schemes with explicit K
    for K in K_VALUES:
        for name, edges in [
            ('uniform', binning.uniform(K)),
            ('quantile', binning.quantile(s_stars, K)),
            ('kmeans_midpoints', binning.kmeans_midpoints(s_stars, K)),
        ]:
            _, ev, ok = fit_evidence(data, len(edges) - 1, edges)
            edges_str = '[' + ', '.join(f'{e:.3f}' for e in edges) + ']'
            ok_str = '' if ok else '  (non-PD)'
            print(f"{name:>20}  {K:>2}  {edges_str:>55}  {ev:>14.2f}{ok_str}")
            results.append((name, K, edges, ev, ok))

    # KDE valleys — K is determined by bandwidth
    for h in KDE_BANDWIDTHS:
        edges = binning.kde_valleys(s_stars, bandwidth=h)
        K_kde = len(edges) - 1
        _, ev, ok = fit_evidence(data, K_kde, edges)
        edges_str = '[' + ', '.join(f'{e:.3f}' for e in edges) + ']'
        ok_str = '' if ok else '  (non-PD)'
        print(f"{'kde_valleys (h=' + f'{h:.2f}' + ')':>20}  {K_kde:>2}  "
              f"{edges_str:>55}  {ev:>14.2f}{ok_str}")
        results.append((f'kde_valleys h={h}', K_kde, edges, ev, ok))

    best = max(results, key=lambda r: r[3])
    print(f"\nBest: {best[0]}, K={best[1]}, log evidence = {best[3]:.2f}")
    print(f"      edges = {best[2].round(3).tolist()}")

    # Plot: bar chart by picker × K, plus s* histogram with best edges
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    by_picker = {}
    for r in results:
        by_picker.setdefault(r[0].split(' ')[0], []).append(r)

    x = np.arange(len(K_VALUES))
    width = 0.25
    pickers_for_bars = ['uniform', 'quantile', 'kmeans_midpoints']
    for i, name in enumerate(pickers_for_bars):
        evs = [next((r[3] for r in results if r[0] == name and r[1] == K), np.nan)
               for K in K_VALUES]
        axes[0].bar(x + (i - 1) * width, evs, width, label=name, color=f'C{i}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'K={K}' for K in K_VALUES])
    axes[0].set_ylabel('Laplace log evidence')
    axes[0].set_title('Bin picker × K (higher is better)')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].set_ylim(top=max(r[3] for r in results) + 1)

    # s* histogram with best picker's edges overlaid
    axes[1].hist(s_stars, bins=20, range=(0, 1), color='C1', alpha=0.6,
                 edgecolor='black', linewidth=0.4, label='observed s*')
    for e in best[2][1:-1]:
        axes[1].axvline(e, color='C3', lw=2, alpha=0.8)
    axes[1].axvline(np.nan, color='C3', lw=2, label=f'best edges ({best[0]}, K={best[1]})')
    axes[1].set_xlabel('s* (fractional position of death)')
    axes[1].set_ylabel('# deaths')
    axes[1].set_title('s* distribution with best edges overlaid')
    axes[1].set_xlim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=110)
    print(f"\nSaved comparison to {SAVE_PATH}")


if __name__ == '__main__':
    main()
