# Changelog

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
