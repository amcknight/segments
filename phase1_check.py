"""Phase 1 -- Static-truth sanity check (the existential gate).

Fits the learning-curve model on test_synth.tsv (static synth, Delta_truth = 0
for all params). The learning model is a strict generalization and should
converge to a flat-curve fit indistinguishable from the static model.

Pass criteria (from external_docs/learning_curves.md):
    - Delta posterior median within +/- 0.5 of 0 for all three params
      (here we only have the MAP point estimate -- skipping Hessian per plan)
    - M_clear matches static fit within ~1s
    - theta_inf MAP entries close to the static-model MAP

MAP-only; no Hessian / no posterior bands -- that's Phase 1 step 2.

Usage:
    python phase1_check.py
    python phase1_check.py --data test_synth.tsv
"""
import argparse
import csv
import time

import numpy as np
from scipy.optimize import minimize

import config
import learning_model
import model
from inference import find_map, _initial_simplex, _bounds


def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append((r['outcome'].strip(), float(r['time_ms'])))
    return rows


def _learning_nll(theta, data, hazard):
    lp = learning_model.log_posterior(theta, data, hazard)
    if not np.isfinite(lp):
        return config.NLL_PENALTY
    return -lp


def find_map_learning(data, hazard, theta_init=None):
    """MAP for the learning model. Mirrors inference.find_map but calls
    learning_model.log_posterior. Bounds reuse inference._bounds (bpt is
    still theta[0])."""
    if theta_init is None:
        theta_init = learning_model.initial_theta(hazard)
    theta_init = theta_init.copy()

    bounds = _bounds(data, len(theta_init))

    bpt_upper = bounds[0][1]
    if theta_init[0] >= bpt_upper:
        theta_init[0] = bpt_upper - config.NM_SIMPLEX_DELTA

    result = minimize(
        _learning_nll,
        theta_init,
        args=(data, hazard),
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'xatol': config.NM_XATOL,
            'fatol': config.NM_FATOL,
            'maxiter': config.NM_MAXITER * 3,    # 10-d, give it room
            'adaptive': True,
            'initial_simplex': _initial_simplex(theta_init),
        },
    )
    return result.x, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='test_synth.tsv')
    args = ap.parse_args()

    data = load_data(args.data)
    hazard = learning_model.haz1()        # K=1 piecewise -- the v1 commitment

    print(f"Loaded {len(data)} attempts from {args.data}")
    n_died = sum(1 for o, _ in data if o == 'died')
    print(f"  {n_died} died, {len(data) - n_died} survived\n")

    # --- Static fit (baseline) -----------------------------------------------
    print("Fitting STATIC model (haz1, d=4)...")
    t0 = time.perf_counter()
    theta_static, result_s = find_map(data, hazard)
    dt_static = time.perf_counter() - t0
    bpt_s, sf_s, ssp_s, haz_s = model.unpack(theta_static, hazard)
    alpha_s = float(haz_s[0])
    m_clear_static = model.m_clear(theta_static, hazard) / 1000.0
    print(f"  done in {dt_static:.1f}s, {result_s.nit} iters, "
          f"converged={result_s.success}")
    print(f"  bpt        = {bpt_s:>10.0f} ms")
    print(f"  slop_frac  = {sf_s:>10.4f}")
    print(f"  slop_sprd  = {ssp_s:>10.4f}")
    print(f"  alpha      = {alpha_s:>10.4f}")
    print(f"  M_clear    = {m_clear_static:>10.2f} s\n")

    # --- Learning fit (cold start) -------------------------------------------
    print("Fitting LEARNING model (haz1 + curves, d=10), cold start...")
    t0 = time.perf_counter()
    theta_learn_cold, result_l_cold = find_map_learning(data, hazard)
    dt_learn = time.perf_counter() - t0
    logpost_cold = learning_model.log_posterior(theta_learn_cold, data, hazard)
    print(f"  done in {dt_learn:.1f}s, {result_l_cold.nit} iters; "
          f"log_post = {logpost_cold:.3f}")

    # --- Learning fit (warm start from static-collapse) ----------------------
    # If the architecture is sound, this point should be a local optimum AND
    # have higher (or equal) log_post than the cold-start point. If the cold-
    # start has a higher posterior, the learning model genuinely prefers a
    # non-flat curve on this nominally-static data.
    print("Fitting LEARNING model, warm start from static-collapse...")
    theta_init_warm = learning_model.initial_theta(hazard)
    theta_init_warm[:4] = theta_static            # asymptotes at static MAP
    # Delta=0, log_tau at prior mean already from initial_theta
    t0 = time.perf_counter()
    theta_learn_warm, result_l_warm = find_map_learning(
        data, hazard, theta_init=theta_init_warm)
    dt_warm = time.perf_counter() - t0
    logpost_warm = learning_model.log_posterior(theta_learn_warm, data, hazard)
    print(f"  done in {dt_warm:.1f}s, {result_l_warm.nit} iters; "
          f"log_post = {logpost_warm:.3f}\n")

    # Also evaluate log_post at the explicit Delta=0 point (using static MAP
    # asymptotes). This is the "is the curve adding evidence?" probe.
    theta_static_collapse = np.empty(learning_model.N_PARAMS)
    theta_static_collapse[:4] = theta_static
    theta_static_collapse[4:7] = 0.0
    theta_static_collapse[7:10] = learning_model.PRIOR_LOG_TAU_MEAN
    logpost_collapse = learning_model.log_posterior(
        theta_static_collapse, data, hazard)
    print(f"  log_post at frozen Delta=0, tau=prior point: {logpost_collapse:.3f}")
    print(f"  log_post gain from curve over static-collapse: "
          f"{max(logpost_cold, logpost_warm) - logpost_collapse:.3f} nats\n")

    # Pick the higher-posterior solution as the canonical learning fit.
    if logpost_warm > logpost_cold:
        print(f"  -> warm start wins by {logpost_warm - logpost_cold:.3f} nats; "
              "using warm result")
        theta_learn, result_l = theta_learn_warm, result_l_warm
    else:
        print(f"  -> cold start wins by {logpost_cold - logpost_warm:.3f} nats; "
              "using cold result")
        theta_learn, result_l = theta_learn_cold, result_l_cold
    print()

    # Asymptotes
    log_bpt = theta_learn[0]
    log_sf_inf = theta_learn[1]
    log_ssp_inf = theta_learn[2]
    log_alpha_inf = theta_learn[3]
    bpt_l = float(np.exp(log_bpt))
    sf_inf = float(np.exp(log_sf_inf))
    ssp_inf = float(np.exp(log_ssp_inf))
    alpha_inf = float(np.exp(log_alpha_inf))

    # Deltas
    delta_sf = float(theta_learn[4])
    delta_ssp = float(theta_learn[5])
    delta_alpha = float(theta_learn[6])

    # Taus
    tau_sf = float(np.exp(theta_learn[7]))
    tau_ssp = float(np.exp(theta_learn[8]))
    tau_alpha = float(np.exp(theta_learn[9]))

    # M_clear at the final attempt -- this is what we compare to the static fit.
    n_final = len(data)
    m_clear_learn_final = learning_model.m_clear_at_n(theta_learn, hazard, n_final) / 1000.0
    m_clear_learn_inf = model.m_clear(theta_learn[:4], hazard) / 1000.0

    print(f"  iters {result_l.nit}, converged={result_l.success}")
    print(f"  bpt           = {bpt_l:>10.0f} ms     (static: {bpt_s:.0f})")
    print(f"  slop_frac_inf = {sf_inf:>10.4f}        (static sf: {sf_s:.4f})")
    print(f"  slop_sprd_inf = {ssp_inf:>10.4f}        (static ssp: {ssp_s:.4f})")
    print(f"  alpha_inf     = {alpha_inf:>10.4f}        (static alpha: {alpha_s:.4f})")
    print(f"  Delta_sf      = {delta_sf:>10.4f}")
    print(f"  Delta_ssp     = {delta_ssp:>10.4f}")
    print(f"  Delta_alpha   = {delta_alpha:>10.4f}")
    print(f"  tau_sf        = {tau_sf:>10.2f} attempts")
    print(f"  tau_ssp       = {tau_ssp:>10.2f} attempts")
    print(f"  tau_alpha     = {tau_alpha:>10.2f} attempts")
    print(f"  M_clear (n={n_final}) = {m_clear_learn_final:>10.2f} s")
    print(f"  M_clear (n=inf) = {m_clear_learn_inf:>10.2f} s\n")

    # --- Gate -----------------------------------------------------------------
    print("Phase 1 pass criteria:")
    DELTA_TOL = 0.5
    MCLEAR_TOL_S = 1.0

    pass_delta_sf = abs(delta_sf) <= DELTA_TOL
    pass_delta_ssp = abs(delta_ssp) <= DELTA_TOL
    pass_delta_alpha = abs(delta_alpha) <= DELTA_TOL
    pass_mclear = abs(m_clear_learn_final - m_clear_static) <= MCLEAR_TOL_S

    def fmt(ok):
        return "PASS" if ok else "FAIL"

    print(f"  |Delta_sf|    <= {DELTA_TOL}:  {fmt(pass_delta_sf)} "
          f"({delta_sf:+.3f})")
    print(f"  |Delta_ssp|   <= {DELTA_TOL}:  {fmt(pass_delta_ssp)} "
          f"({delta_ssp:+.3f})")
    print(f"  |Delta_alpha| <= {DELTA_TOL}:  {fmt(pass_delta_alpha)} "
          f"({delta_alpha:+.3f})")
    print(f"  |M_clear diff| <= {MCLEAR_TOL_S}s: {fmt(pass_mclear)} "
          f"(learning {m_clear_learn_final:.2f}s vs static {m_clear_static:.2f}s, "
          f"diff {m_clear_learn_final - m_clear_static:+.2f}s)")

    all_pass = all([pass_delta_sf, pass_delta_ssp, pass_delta_alpha, pass_mclear])
    print(f"\n  Overall: {'PASS -- existential test cleared' if all_pass else 'FAIL -- architecture review needed'}")


if __name__ == '__main__':
    main()
