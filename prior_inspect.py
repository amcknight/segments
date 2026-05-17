"""Inspect the learning model at N=0 and N=1.

N=0: log_posterior = log_prior. The MAP is the joint prior mode -- for our
priors (all Normal in unconstrained space) that's just the prior means. We
trace out what the prior implies for theta(n) and M_clear over n, and how
prior-1-std perturbations on each latent change the curve.

N=1: refit with a single attempt to see how much one data point can move
things. Both the survived and the died case.

The question this answers: are our starting priors sensible, or do they
imply an obviously-wrong player at n=0 that 150 attempts can't pull away from?

Usage:
    python prior_inspect.py
"""
import numpy as np

import config
import learning_model
import model
from generate_synthetic_learning import TRUTH_DEFAULTS
from phase2_recovery import m_clear_truth_at_n, theta_to_natural


def prior_mode_theta(hazard):
    """Joint mode of the priors: all Normal, so means."""
    return learning_model.initial_theta(hazard)


def theta_at_n_natural(theta, n):
    """Get [sf(n), ssp(n), alpha_haz(n), P(die)(n)] at attempt n."""
    static = learning_model.static_theta_at_n(theta, n)
    sf_n = float(np.exp(static[1]))
    ssp_n = float(np.exp(static[2]))
    alpha_haz_n = float(np.exp(static[3]))
    p_die_n = 1.0 - np.exp(-alpha_haz_n)
    return sf_n, ssp_n, alpha_haz_n, p_die_n


def section(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main():
    hazard = learning_model.haz1()
    truth = TRUTH_DEFAULTS

    section("N=0 -- prior mode (joint prior MAP, no data)")
    theta_prior = prior_mode_theta(hazard)
    print(f"Prior-mode parameters in natural units:")
    nat = theta_to_natural(theta_prior)
    print(f"  bpt           = {nat[0]:>10.0f} ms")
    print(f"  sf_inf        = {nat[1]:>10.4f}  (prior median for slop_frac)")
    print(f"  ssp_inf       = {nat[2]:>10.4f}  (prior median for slop_spread)")
    print(f"  alpha_haz_inf = {nat[3]:>10.4f}  (prior median for haz_1)")
    print(f"  Delta_sf      = {nat[4]:>10.4f}  (Normal(0,2) prior)")
    print(f"  Delta_ssp     = {nat[5]:>10.4f}")
    print(f"  Delta_alpha   = {nat[6]:>10.4f}")
    print(f"  tau_sf        = {nat[7]:>10.2f}  (prior median for tau)")
    print(f"  tau_ssp       = {nat[8]:>10.2f}")
    print(f"  tau_alpha     = {nat[9]:>10.2f}")

    print(f"\nPrior-implied per-attempt theta(n) (at prior mode all Deltas = 0,")
    print(f"so theta(n) = theta_inf for all n -- flat curve):")
    print(f"  {'n':>4}  {'sf(n)':>8}  {'ssp(n)':>8}  {'alpha_h(n)':>10}  "
          f"{'P(die)':>8}  {'M_clear(s)':>10}")
    for n in [1, 5, 25, 50, 150, 500]:
        sf_n, ssp_n, ah, pd = theta_at_n_natural(theta_prior, n)
        mc = learning_model.m_clear_at_n(theta_prior, hazard, n) / 1000.0
        print(f"  {n:>4}  {sf_n:>8.4f}  {ssp_n:>8.4f}  {ah:>10.4f}  "
              f"{pd:>8.4f}  {mc:>10.2f}")

    section("Prior-1-std excursions on each latent (one at a time)")
    print(f"How much does theta(1) and M_clear(1) move when each latent is")
    print(f"perturbed +/- 1 prior std from its prior mean?")
    print()
    print(f"  {'latent':>14}  {'mean':>8}  {'+1sigma':>8}  {'-1sigma':>8}  "
          f"{'M(+1s)':>8}  {'M(-1s)':>8}")
    excursion_specs = [
        ('log bpt',       0, config.PRIOR_LOG_BPT_SD),
        ('log sf_inf',    1, config.PRIOR_LOG_SLOP_FRAC_SD),
        ('log ssp_inf',   2, config.PRIOR_LOG_SLOP_SPREAD_SD),
        ('log alpha_inf', 3, config.PRIOR_LOG_HAZ_SD),
        ('Delta_sf',      4, learning_model.PRIOR_DELTA_SD),
        ('Delta_ssp',     5, learning_model.PRIOR_DELTA_SD),
        ('Delta_alpha',   6, learning_model.PRIOR_DELTA_SD),
        ('log tau_sf',    7, learning_model.PRIOR_LOG_TAU_SD),
        ('log tau_ssp',   8, learning_model.PRIOR_LOG_TAU_SD),
        ('log tau_alpha', 9, learning_model.PRIOR_LOG_TAU_SD),
    ]
    for label, i, sd in excursion_specs:
        m_plus = learning_model.m_clear_at_n(
            np.array([t + (sd if j == i else 0) for j, t in enumerate(theta_prior)]),
            hazard, n=1) / 1000.0
        m_minus = learning_model.m_clear_at_n(
            np.array([t - (sd if j == i else 0) for j, t in enumerate(theta_prior)]),
            hazard, n=1) / 1000.0
        print(f"  {label:>14}  {theta_prior[i]:>+8.3f}  "
              f"{theta_prior[i]+sd:>+8.3f}  {theta_prior[i]-sd:>+8.3f}  "
              f"{m_plus:>8.2f}  {m_minus:>8.2f}")

    section("Truth comparison: prior says vs truth says")
    theta_truth = np.array([
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
    print(f"  {'n':>4}  {'prior M_c(s)':>12}  {'truth M_c(s)':>12}  "
          f"{'ratio':>8}")
    for n in [1, 5, 25, 50, 150, 500]:
        m_prior = learning_model.m_clear_at_n(theta_prior, hazard, n) / 1000.0
        m_truth = m_clear_truth_at_n(truth, hazard, n) / 1000.0
        print(f"  {n:>4}  {m_prior:>12.2f}  {m_truth:>12.2f}  "
              f"{m_prior/m_truth:>8.2f}")

    section("N=1 -- fit on a single attempt")
    from phase1_check import find_map_learning

    # Use one survived and one died "typical" attempt from truth.
    typical_survived = ('survived', truth['bpt'] + 0.05 * truth['bpt'])  # ~5% over bpt
    typical_died = ('died', 0.5 * truth['bpt'] + config.RESPAWN_MS)      # died at half clean time

    for label, datum in [('1 survived', [typical_survived]),
                          ('1 died', [typical_died])]:
        print(f"\nFit on {label}: {datum}")
        theta_map, result = find_map_learning(datum, hazard)
        nat = theta_to_natural(theta_map)
        print(f"  {result.nit} iters, converged={result.success}")
        for i, label_p in enumerate([
                'bpt(ms)', 'sf_inf', 'ssp_inf', 'a_h_inf',
                'D_sf', 'D_ssp', 'D_alpha',
                'tau_sf', 'tau_ssp', 'tau_alpha']):
            move = theta_map[i] - theta_prior[i]
            print(f"    {label_p:>10} = {nat[i]:>10.4f}  "
                  f"(prior {theta_to_natural(theta_prior)[i]:.4f}, "
                  f"d_log = {move:+.4f})")


if __name__ == '__main__':
    main()
