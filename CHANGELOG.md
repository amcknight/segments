# Changelog

## 2026-05-17 — PPC, NUTS, full halflife pool, smooth-abs bugfix, fit_core refactor, cleanup

Major session. Completed steps 1-3 of plan_forward_sequence; deferred step 4
(API contract) until shape-param learning is settled.

### New capability

- **PPC infrastructure** ([ppc.py](ppc.py)): vectorized forward simulator,
  4 discrepancy stats (died_rate, tau skew/kurt, mid-third), mid-p Bayes
  p-value. 8 tests in [tests/test_ppc.py](tests/test_ppc.py). 2×2 sanity
  check ([tmp/ppc_synth_check.py](tmp/ppc_synth_check.py)) confirms haz1-
  on-beta2-truth caught at p≈0.008 on the mid-third stat with no false
  alarms on the other three cells.
- **NUTS via NumPyro** ([fit_nuts.py](fit_nuts.py)): wraps the JAX
  `log_posterior` via `ImproperUniform + factor`. `posterior_samples()` is
  the auto-fallback (Laplace where PD, NUTS otherwise). 7 tests in
  [tests/test_nuts.py](tests/test_nuts.py). Bench: NUTS 23-453s depending
  on N and model; Laplace stays <2s.
- **Full halflife pool** ([fit_eb_pool.py](fit_eb_pool.py)): `Pool` extended
  from 2→6 fields covering sf/ssp/alpha × mean/sigma. All priors and fit
  modules accept 6 pool kwargs symmetrically. 5 new tests covering ssp/
  alpha population recovery, regime-1 shrinkage, and multi-pool
  independence.

### Bugfixes

- **|A| Hessian autodiff bug** (`halflife_sigma`): `jnp.abs(A)` corrupted
  the second derivative at A=0 (the no-learning MAP). Caused 4/5 cells in
  the NUTS bench to fail Laplace PD on well-spec data. Replaced with
  `smooth_abs(A) = sqrt(A² + ε²) - ε`. Documented in
  [external_docs/pgm_tricks.md](external_docs/pgm_tricks.md) "Smooth-once
  isn't smooth-twice". Regression test in
  [tests/test_joint_prior.py](tests/test_joint_prior.py) verified to fail
  when reintroduced.
- **EB σ collapse**: iterative both-parameter EB algorithm pegged σ_pop at
  the floor (0.05) regardless of true population spread, via the feedback
  loop σ↓ → MAPs↓ → std(MAPs)↓. Switched to one-step σ (estimated once
  from broad-prior MAPs) + iterated μ. Honest SD-reduction numbers replace
  the inflated 68-86% figures. Documented in pgm_tricks.md "EB σ collapse
  — iterate the mean, fix the scale".

### Architecture

- **[fit_core.py](fit_core.py)**: generic fit machinery parametrized over
  the model module. `fit_jax.py` and `fit_jax_beta2.py` are now thin
  shims (warm_init + binding). Adding a third hazard model is ~30-50
  lines. JIT cache stays tight via `lru_cache` on the model identity.
- **Public API surface** in `segments_v07.py` updated: `Pool` now exposes
  6 fields with `as_kwargs()` convenience.

### Calibration probes (new diagnostic infrastructure)

[tmp/calibration_probes.py](tmp/calibration_probes.py) — four self-
consistency probes enabled by PPC + NUTS:
1. Hessian autodiff vs FD agreement — would have caught the |A| bug
   (8/8 seeds clean post-fix).
2. NUTS-vs-Laplace SD agreement across many seeds.
3. Well-spec PPC p-value distribution — shape stats ~uniform (KS 0.10-0.15).
4. **Posterior coverage** — 90% NUTS interval contains truth across 20
   seeds. Result: 9/10 dims ≥0.85; `log_hl_ssp` at 0.70 (under-coverage
   from joint-prior collapse × ssp prior/truth offset; documented in
   memory/coverage-finding.md — known design trade-off, not a bug).

### Cleanup

- Deleted 12 pre-V07 files (recoverable via git): `learning_model.py`,
  `generate_synthetic*.py`, `phase1_check.py`, `phase2_recovery.py`,
  `pgm_inspect.py`, `pooling_experiment.py`, `prior_inspect.py`,
  `bin_search.py`, `evidence_scan.py`, `preview_synthetic.py`, `run.py`.
  All were either pre-V07 with V07 successors, or referenced model
  parameterizations the V07 lineage doesn't use. Pre-deletion grep
  confirmed no current file imports any of them.
- Old `_laplace_sds_for_pooled` helper removed from fit_eb_pool (dead
  code after the σ-collapse fix made deconvolution unnecessary).

### Headline numbers

- Tests: 45 passing (was 24 before this session). Suite runs in ~90s.
- All 5 cells in the NUTS bench are Laplace PD post-smooth-abs fix
  (was 1/5 pre-fix). NUTS R-hat improved across the board (e.g. beta2
  N=100: 1.053 → 1.019).
- EB pool: 29-57% per-segment SD reduction on the three pooled halflives
  for 6 hyperparams of cost (the old buggy 68-86% was inflated).
- Pool means actually drift toward truth on synth: sf +84% of gap, ssp
  +54%, alpha +79%.

## 2026-05-16 (bonus session) — Empirical-Bayes pool on halflife_sf

## 2026-05-16 (bonus session) — Empirical-Bayes pool on halflife_sf

The "smallest principled HYPER" step landed.

- [fit_eb_pool.py](fit_eb_pool.py): `Pool` namedtuple + `fit_eb_pool()` driver
  + `fit_independent()` baseline. Iterates per-segment fits ↔ pool re-estimation
  until convergence (typically 3-8 iters, `tol=1e-3` by default).
- [generate_synthetic_learning_v07.py](generate_synthetic_learning_v07.py):
  added `generate_multi_segment(n_segments, ..., true_pop_log_halflife_sf_mean,
  ..._sigma, ...)` for population-truth synth.
- Refactored `log_prior` / `log_posterior` / `find_map` /
  `laplace_posterior` (both numpy and JAX) to accept
  `halflife_sf_pop_mean=` / `halflife_sf_pop_sigma=` overrides. Defaults
  unchanged → all prior tests still pass.
- 4 new tests in [tests/test_eb_pool.py](tests/test_eb_pool.py): population
  recovery, convergence, regime-1 shrinkage, single-segment degeneracy.
- Public API surface in `segments_v07.py` extended: `sv.Pool`,
  `sv.fit_eb_pool`, `sv.fit_independent` (29 symbols total).

**Headline numbers** (5 segments at N=300; 4 informative, 1 regime-1):
- Independent fits: 413 ms total (≈83 ms/segment, post-JIT-warmup).
- EB pooled fits (5-8 iters, each iter warm-starts from prev MAP): 1049 ms
  total (≈210 ms/segment, ≈26 ms per warm-started fit).
- **Regime-1 segment's halflife: 29 attempts (independent, ≈ prior median)
  → 45 attempts (pooled, = population mean).** Distance to pop target
  collapses from 15.8 to 0.
- Pool converges in ~5 iterations with `tol=1e-3`.

The EB loop is ~2.5× the cost of independent fits at the segment count
that justifies pooling (≥5). At scale (100+ segments) the multiplier
stays in the 3-5× range because the iteration count doesn't grow with S.
Production pattern: re-fit pool periodically (when 10% new data
accumulates), not on every attempt.

Test count: **24/24 passing in 10.8s.**

---

## 2026-05-16 — V07 halflife reparam, joint prior, JAX rewrite

Big session. Closed out Plan V07 + the model-lock-down work needed before
production handoff.

### What landed

**Model**
- Halflife reparameterization: `tau` → `halflife` everywhere, curve form
  switched to `2^(-(n-1)/halflife)`. Synth byte-identical to v0.6 at the
  same seed (purely a rename + base change).
- Joint prior `P(halflife | A)` via `tanh`-smoothed sigma. Collapses
  regime-1 halflife from 8–48 (6× spread) to a single grid cell at the
  prior median when there's no learning to characterize.

**JAX path** (the big speed unlock)
- [learning_model_v07_jax.py](learning_model_v07_jax.py): JAX rewrite of
  the V07 model. Validated against numpy to ~1e-5 on log_posterior.
- [fit_jax.py](fit_jax.py): `find_map`, `warm_init_from_data`,
  `prewarm_buckets`, `laplace_posterior`, `sample_laplace`. Single
  bucketed/padded entry point.
- **~700× speedup on cold MAP fits (70s → 100ms), p50 8ms on streaming
  refits (100% sub-200ms).** See [external_docs/speed_and_techniques.md](external_docs/speed_and_techniques.md).

**API surface**
- [segments_v07.py](segments_v07.py): single-import public API
  (`import segments_v07 as sv`). 26 exported symbols.
- [README.md](README.md): file map, quick start, perf table, dev workflow.

**Tests**
- [tests/](tests/): 20 pytest tests across 4 files. ~7s end-to-end. Cover
  curve identities, JAX≡numpy parity, joint-prior behavior, fit pipeline.

**Docs added**
- [external_docs/pgm_tricks.md](external_docs/pgm_tricks.md): glossary of
  PGM patterns (ridges, joint priors, partial pooling, Neal funnel, EB).
- [external_docs/speed_and_techniques.md](external_docs/speed_and_techniques.md):
  this session's speed report + technique list + first-hyperprior estimate.

**Infrastructure**
- `.venv/` with `requirements.txt` (numpy, scipy, jax, jaxopt, pytest).
- `data/db.py` + `data/schema.sql`: minimal sqlite layer for attempt logs.

### Open items (deferred, picked in priority order)

1. **One-hyperprior empirical-Bayes pool on `halflife_sf`.** 1–2 sessions.
   Rationale and lift estimate in `speed_and_techniques.md`. Smallest
   principled HYPER step.
2. **NUTS posterior fallback** for ridge cases (BlackJAX or NumPyro).
   Laplace fails when Hessian goes non-PD.
3. **Validation harness with golden TSV/JSON** for the receiving Python
   project to verify its port matches.
4. **TypeScript schema** for the live-viz consumer (bands + MAP + metadata).
5. **GPU JAX + `jax.vmap` batched per-segment fits** for the
   thousands-of-segments target. Single library upgrade; the model code
   doesn't change.
6. **Recording pipeline / UI**: data layer exists, no inserter yet.

### Where to resume

Read order for a cold context:
1. [README.md](README.md) — file map and where things live.
2. [external_docs/speed_and_techniques.md](external_docs/speed_and_techniques.md) — what was learned and what's next.
3. [external_docs/learning_curves_session_findings.md](external_docs/learning_curves_session_findings.md) — strategic plan.
4. [external_docs/pgm_tricks.md](external_docs/pgm_tricks.md) — vocabulary for the design choices.

Run `.venv/Scripts/python -m pytest tests/ -v` to confirm everything green.
