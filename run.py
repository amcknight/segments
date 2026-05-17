"""Stream attempts from test_data.tsv, run MAP inference per attempt, plot.

After streaming, computes an end-of-session Laplace covariance at the final
MAP and uses posterior samples to draw uncertainty bands on the survival
curve and haz_k marginal histograms.

Usage:
    python run.py            # fast: no real-time sleeps
    python run.py --realtime # wait time_ms between attempts
"""
import argparse
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

import config
import binning
from hazard import PiecewiseConstantHazard, BetaMixtureHazard
from model import (
    unpack, lognormal_params, clean_dist_pdf, m_clear,
    slop_moments,
)
from inference import find_map, laplace_cov, sample_posterior


# Build the survival model used by this script. Swap via --model on the CLI;
# any HazardModel works downstream because the likelihood / MAP / Laplace code
# only sees the abstract interface.
def hazard_for(model, K):
    if model == 'piecewise':
        return PiecewiseConstantHazard(K, binning.uniform(K))
    if model == 'beta':
        return BetaMixtureHazard(K)
    raise ValueError(f"unknown model {model!r} (expected 'piecewise' or 'beta')")


# Run-time settings
# -----------------
DATA_PATH = 'test_data.tsv'
SCALAR_PARAM_NAMES = ['bpt (ms)', 'slop_frac', 'slop_spread']

# Plot settings
# -------------
PP_GRID_LO_FRAC = 0.9              # clean_dist plot: lower x-bound as fraction of bpt
PP_GRID_HI_SLOP_MULT = 6.0         # clean_dist plot: upper x-bound = bpt + this * mean_slop
PP_GRID_POINTS = 400               # number of T points in the predictive density plot
PP_GRID_FLOOR_MS = 1000            # safety floor if bpt * PP_GRID_LO_FRAC is unreasonably small
PP_HIST_BINS = 8                   # bins for the survived-times histogram

SURV_GRID_POINTS = 200             # number of points along the time-axis survival curve

# M_clear y-axis: the prior-dominated first few attempts can produce huge
# M_clear values that compress the converged region into a flat line at the
# bottom. We exclude these warmup attempts when picking the y upper bound; the
# data points are still drawn but clip out the top of the chart.
M_CLEAR_WARMUP_ATTEMPTS = 5
M_CLEAR_YLIM_HEADROOM = 1.2        # multiplier above post-warmup max

FIG_WIDTH = 14
FIG_HEIGHT = 9
FIG_DPI = 110


def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append((r['outcome'].strip(), float(r['time_ms'])))
    return rows


def _hazard_param_labels(hazard):
    """Short column labels for the hazard's theta-slice entries. Returned in
    the same order they appear in theta (so labels line up with np.exp(theta)).
    """
    if isinstance(hazard, PiecewiseConstantHazard):
        return [f'haz_{k+1}' for k in range(hazard.K)]
    if isinstance(hazard, BetaMixtureHazard):
        K = hazard.K
        labels = ['alpha']
        labels += [f'mu_{k+1}' for k in range(K)]
        labels += [f'kap_{k+1}' for k in range(K)]
        labels += [f'eta_{k+1}' for k in range(K - 1)]   # last weight is reference
        return labels
    return [f'p_{i}' for i in range(hazard.n_params)]


def _hazard_param_display(hazard, theta_slice):
    """How to display each hazard param in the streaming row.

    Most params we show in original (exp-of-logspace) units; alpha and mu_k are
    logits so we sigmoid them. Returned values line up 1:1 with the labels.
    """
    if isinstance(hazard, BetaMixtureHazard):
        K = hazard.K
        t = np.asarray(theta_slice)
        alpha_disp = float(1.0 / (1.0 + np.exp(-t[0])))
        mu_disp = (1.0 / (1.0 + np.exp(-t[1:1 + K]))).tolist()
        kap_disp = np.exp(t[1 + K:1 + 2 * K]).tolist()
        eta_disp = t[1 + 2 * K:3 * K].tolist()           # softmax logits, raw
        return [alpha_disp] + mu_disp + kap_disp + eta_disp
    return np.exp(theta_slice).tolist()


def stream(data_all, hazard, realtime=False):
    seen = []
    theta_prev = None
    history = []

    haz_labels = _hazard_param_labels(hazard)
    header = f"{'#':>3} {'outcome':>9} {'time':>6}"
    for n in SCALAR_PARAM_NAMES + haz_labels:
        header += f"  {n:>10}"
    header += "  iters    ms"
    print(header)

    for i, (outcome, time_ms) in enumerate(data_all, 1):
        if realtime:
            time.sleep(time_ms / 1000.0)
        seen.append((outcome, time_ms))

        t0 = time.perf_counter()
        theta_map, result = find_map(seen, hazard, theta_init=theta_prev)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        theta_prev = theta_map

        scalar_vals = np.exp(theta_map[:3]).tolist()
        haz_vals = _hazard_param_display(hazard, theta_map[3:3 + hazard.n_params])
        row = f"{i:>3} {outcome:>9} {time_ms:>6.0f}"
        for v in scalar_vals + haz_vals:
            row += f"  {v:>10.4g}"
        row += f"  {result.nit:>5d}  {dt_ms:>4.0f}"
        print(row)
        history.append({'theta_map': theta_map.copy()})

    return history, seen


def plot_m_clear(history, hazard, seen, ax):
    """Expected total time to clear, evaluated at each step's MAP."""
    m = np.array([m_clear(h['theta_map'], hazard) for h in history]) / 1000.0
    xs = np.arange(1, len(history) + 1)
    ax.plot(xs, m, color='C3', lw=2.0, marker='.')
    ax.set_xlabel('attempt #')
    ax.set_ylabel('expected total clear time (s)')
    ax.set_title('M_clear: expected wall-clock to first survival, including resets')
    ax.grid(alpha=0.3)
    final = m[-1]
    ax.axhline(final, color='C3', ls='--', lw=0.8, alpha=0.6,
               label=f'final estimate: {final:.1f}s')
    ax.legend(loc='upper right')

    if len(m) > M_CLEAR_WARMUP_ATTEMPTS:
        ylim_top = float(m[M_CLEAR_WARMUP_ATTEMPTS:].max()) * M_CLEAR_YLIM_HEADROOM
        ax.set_ylim(0, ylim_top)


def _derived_series(history, hazard):
    """Per-attempt interpretable quantities derived from MAP. Model-agnostic
    portion only: bpt, mean/std clean time, P(die per attempt)."""
    bpts, means, stds, p_dies = [], [], [], []
    for h in history:
        theta = h['theta_map']
        bpt, sf, ssp, haz_params = unpack(theta, hazard)
        mu_log, sigma_log = lognormal_params(sf, ssp, bpt)
        mean_slop, std_slop = slop_moments(mu_log, sigma_log)
        bpts.append(bpt / 1000.0)
        means.append((bpt + mean_slop) / 1000.0)
        stds.append(std_slop / 1000.0)
        p_dies.append(1.0 - float(np.exp(-hazard.cum(1.0, haz_params))))
    return (np.array(bpts), np.array(means), np.array(stds), np.array(p_dies))


def plot_param_trajectories(history, hazard, axes):
    """4 panels in user-facing units:
        (1) best clean time (bpt)
        (2) mean clean time ± 1 std
        (3) probability of dying per attempt
        (4) model-specific 'where the death mass is' trajectory:
              piecewise -> per-bin hazard rate
              beta      -> per-component mean mu_k
    """
    bpts, means, stds, p_dies = _derived_series(history, hazard)
    xs = np.arange(1, len(history) + 1)
    flat = axes.flat

    flat[0].plot(xs, bpts, color='C0', lw=1.5, marker='.')
    flat[0].set_title('Best clean time, bpt (s)')
    flat[0].set_xlabel('attempt #')
    flat[0].grid(alpha=0.3)

    flat[1].fill_between(xs, means - stds, means + stds, alpha=0.2, color='C0',
                          label='±1 std')
    flat[1].plot(xs, means, color='C0', lw=1.5, marker='.', label='mean')
    flat[1].plot(xs, bpts, color='C1', lw=1.0, alpha=0.6, label='bpt')
    flat[1].set_title('Clean run time, mean ± std (s)')
    flat[1].set_xlabel('attempt #')
    flat[1].legend(loc='upper right', fontsize=8)
    flat[1].grid(alpha=0.3)

    flat[2].plot(xs, p_dies, color='C3', lw=1.5, marker='.')
    flat[2].set_title('P(die per attempt)')
    flat[2].set_xlabel('attempt #')
    flat[2].set_ylim(0, 1)
    flat[2].grid(alpha=0.3)

    _plot_hazard_specific_trajectory(history, hazard, flat[3])


def _plot_hazard_specific_trajectory(history, hazard, ax):
    K = hazard.K
    xs = np.arange(1, len(history) + 1)
    if isinstance(hazard, PiecewiseConstantHazard):
        hazs = np.array([np.exp(h['theta_map'][3:3 + K]) for h in history])
        for k in range(K):
            ax.plot(xs, hazs[:, k], lw=1.5, marker='.',
                    label=f'haz_{k+1}', color=f'C{k+2}')
        ax.set_title(f'Hazard rates per bin, MAP trajectory (K={K})')
        ax.set_ylabel('hazard rate')
    elif isinstance(hazard, BetaMixtureHazard):
        mus = np.array([
            1.0 / (1.0 + np.exp(-h['theta_map'][4:4 + K]))     # 4 = 3 scalars + alpha
            for h in history
        ])
        for k in range(K):
            ax.plot(xs, mus[:, k], lw=1.5, marker='.',
                    label=f'mu_{k+1}', color=f'C{k+2}')
        ax.set_title(f'Component means mu_k, MAP trajectory (K={K})')
        ax.set_ylabel('mu_k (fractional position)')
        ax.set_ylim(0, 1)
    ax.set_xlabel('attempt #')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)


def plot_haz_posterior(samples, hazard, ax):
    """Marginal posterior on the hazard model's interesting parameters."""
    if samples is None:
        ax.text(0.5, 0.5, 'Laplace failed', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('hazard posterior (Laplace)')
        return
    K = hazard.K
    if isinstance(hazard, PiecewiseConstantHazard):
        haz_samples = np.exp(samples[:, 3:3 + K])
        for k in range(K):
            ax.hist(haz_samples[:, k], bins=40, density=True, alpha=0.5,
                    color=f'C{k+2}', label=f'haz_{k+1}')
        ax.set_xlabel('hazard rate')
    elif isinstance(hazard, BetaMixtureHazard):
        # alpha and per-component mu_k are most interpretable.
        alpha_s = 1.0 / (1.0 + np.exp(-samples[:, 3]))
        mu_s = 1.0 / (1.0 + np.exp(-samples[:, 4:4 + K]))
        ax.hist(alpha_s, bins=40, density=True, alpha=0.6,
                color='k', label='alpha = P(die)')
        for k in range(K):
            ax.hist(mu_s[:, k], bins=40, density=True, alpha=0.5,
                    color=f'C{k+2}', label=f'mu_{k+1}')
        ax.set_xlabel('value (logit-transformed back)')
        ax.set_xlim(0, 1)
    ax.set_title('hazard posterior (Laplace samples)')
    ax.set_ylabel('density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)


def plot_posterior_predictive(history, seen, hazard, ax):
    theta_map = history[-1]['theta_map']
    bpt, sf, ssp, haz = unpack(theta_map, hazard)
    mu_log, sigma_log = lognormal_params(sf, ssp, bpt)

    grid_lo = max(PP_GRID_FLOOR_MS, bpt * PP_GRID_LO_FRAC)
    grid_hi = bpt + PP_GRID_HI_SLOP_MULT * (sf * bpt)
    T_grid = np.linspace(grid_lo, grid_hi, PP_GRID_POINTS)
    density = np.array([clean_dist_pdf(t, bpt, mu_log, sigma_log) for t in T_grid])

    survived_times = [t for o, t in seen if o == 'survived']
    ax.hist(survived_times, bins=PP_HIST_BINS, density=True, color='C1', alpha=0.4,
            label=f'survived (n={len(survived_times)})')
    ax.plot(T_grid, density, color='C0', lw=1.5, label='clean_dist at MAP')
    ax.axvline(bpt, color='k', ls='--', lw=1, label=f'bpt = {bpt:.0f}ms')
    ax.set_xlabel('T_clean (ms)')
    ax.set_ylabel('density')
    ax.set_title('clean_dist (MAP) vs survived times')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)


def plot_survival_curve(history, seen, hazard, samples, ax):
    """Survival probability over wall-clock time within an attempt, with
    Laplace bands when posterior samples are available."""
    theta_map = history[-1]['theta_map']
    bpt, sf, ssp, haz_params = unpack(theta_map, hazard)
    mu_log, sigma_log = lognormal_params(sf, ssp, bpt)
    mean_slop, _ = slop_moments(mu_log, sigma_log)
    mean_clean_ms = bpt + mean_slop
    mean_clean_s = mean_clean_ms / 1000.0

    t_grid_s = np.linspace(0.0, mean_clean_s, SURV_GRID_POINTS)
    s_grid = (t_grid_s * 1000.0) / mean_clean_ms

    if samples is not None:
        curves = np.empty((len(samples), len(s_grid)))
        for i, theta in enumerate(samples):
            _, _, _, h_s = unpack(theta, hazard)
            curves[i] = np.exp(-hazard.cum(s_grid, h_s))
        lo50, med, hi50 = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
        lo90, hi90 = np.quantile(curves, [0.05, 0.95], axis=0)
        ax.fill_between(t_grid_s, lo90, hi90, color='C2', alpha=0.15, label='90% CI')
        ax.fill_between(t_grid_s, lo50, hi50, color='C2', alpha=0.30, label='50% CI')
        ax.plot(t_grid_s, med, color='C2', lw=2.0, label='posterior median')
    else:
        surv = np.exp(-hazard.cum(s_grid, haz_params))
        ax.plot(t_grid_s, surv, color='C2', lw=2.0, label='P(alive at t), MAP')

    death_taus_s = [(t - config.RESPAWN_MS) / 1000.0 for o, t in seen if o == 'died']
    for tau in death_taus_s:
        ax.axvline(tau, color='k', alpha=0.3, lw=0.7)
    # Piecewise-specific: bin edges only exist for that model.
    bin_edges = getattr(hazard, 'bin_edges', None)
    if bin_edges is not None:
        for be in bin_edges[1:-1]:
            ax.axvline(be * mean_clean_s, color='C2', alpha=0.4, ls=':', lw=0.8)

    ax.set_xlabel('time into attempt (s)')
    ax.set_ylabel('P(still alive)')
    ax.set_title('Survival curve; black ticks = observed deaths')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--realtime', action='store_true',
                    help='wait time_ms between attempts (real-time replay)')
    ap.add_argument('--save', default='pic/diagnostics.png',
                    help='where to save the diagnostic figure')
    ap.add_argument('--model', choices=['piecewise', 'beta'], default='beta',
                    help='survival model: beta mixture (default) or piecewise hazard')
    ap.add_argument('-K', '--K', type=int, default=None,
                    help='hazard bins (piecewise) or beta-mixture components '
                         '(beta). Default depends on --model: 3 for piecewise, '
                         '2 for beta.')
    args = ap.parse_args()
    if args.K is None:
        args.K = 2 if args.model == 'beta' else 3

    data = load_data(DATA_PATH)
    hazard = hazard_for(args.model, args.K)
    print(f"Loaded {len(data)} attempts. Streaming "
          f"(model={args.model}, K={args.K}, "
          f"respawn={config.RESPAWN_MS:.0f}ms)...")
    history, seen = stream(data, hazard, realtime=args.realtime)

    theta_final = history[-1]['theta_map']
    print("Computing end-of-session Laplace covariance...")
    cov, ok = laplace_cov(theta_final, seen, hazard)
    print(f"  Hessian {'positive-definite' if ok else 'NOT pos-def (using pseudoinverse, bands approximate)'}")
    samples = (sample_posterior(theta_final, cov, config.LAPLACE_N_SAMPLES)
               if cov is not None else None)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT + 5))
    gs = fig.add_gridspec(5, 2)
    m_ax = fig.add_subplot(gs[0, :])
    param_axes = np.array([
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ])
    haz_post_ax = fig.add_subplot(gs[3, 0])
    surv_ax = fig.add_subplot(gs[3, 1])
    pp_ax = fig.add_subplot(gs[4, :])

    plot_m_clear(history, hazard, seen, m_ax)
    plot_param_trajectories(history, hazard, param_axes)
    plot_haz_posterior(samples, hazard, haz_post_ax)
    plot_survival_curve(history, seen, hazard, samples, surv_ax)
    plot_posterior_predictive(history, seen, hazard, pp_ax)

    fig.suptitle(f'Streaming MAP fit; model={args.model}, K={args.K}, '
                 f'N={len(data)} attempts')
    fig.tight_layout()
    fig.savefig(args.save, dpi=FIG_DPI)
    print(f"\nSaved figure to {args.save}")


if __name__ == '__main__':
    main()
