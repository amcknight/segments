# segments — V07 speedrun segment model

Probabilistic model for per-segment speedrun grinding: given a sequence of
attempts (`survived` / `died`, times in ms), infer the underlying skill
parameters and how they evolve as the player learns the segment. Designed
for handoff to a production Python service feeding a TypeScript live viz.

This README covers the file map, how to run things, and where the
formal specs live.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# Sanity-check the install
.venv/Scripts/python -m pytest tests/         # 20 tests, ~7s

# Render the inspector (numpy reference; opens HTML in tmp/)
.venv/Scripts/python pgm_inspect_v07.py --out tmp/v07.html

# Fit on synth (JAX path, fast)
.venv/Scripts/python -c "import segments_v07 as sv; \
  from generate_synthetic_learning_v07 import TRUTH_DEFAULTS, generate; \
  d = generate(500, TRUTH_DEFAULTS, seed=0); \
  t, dd = sv.data_to_arrays(d); \
  th, info = sv.find_map(t, dd); \
  print('MAP iter', info['iter'], 'nll', round(info['nll'],1))"
```

## File map

### Production V07 (the keep-pile)

| File | What |
|---|---|
| `segments_v07.py` | **Public API.** Imports the right things from below; `import segments_v07 as sv` and use. |
| `learning_model_v07_jax.py` | **JAX impl of model + priors.** Differentiable, JIT-compilable. The canonical fitting implementation. |
| `learning_model_v07.py` | **Numpy reference impl.** Slower; used for cross-validation and offline tools (inspector, ridge demos). The tests pin JAX ≡ numpy to ~1e-5. |
| `fit_jax.py` | `find_map`, `warm_init_from_data`, `prewarm_buckets`, `laplace_posterior`, `sample_laplace`. Accepts optional `halflife_sf_pop_*` overrides from an EB pool. |
| `fit_eb_pool.py` | **Empirical-Bayes pool on `halflife_sf`** (HYPER-lite). `Pool` namedtuple + `fit_eb_pool()` iteration driver + `fit_independent()` baseline. |
| `generate_synthetic_learning_v07.py` | Synth generator. V07-shape truth, byte-identical TSV to v0.6 at same seed. Also `generate_multi_segment()` for population-truth multi-segment synth. |
| `phase2_recovery_v07.py` | Recovery experiment runner (numpy fit). The "does V07 hit pass criteria" gate. |
| `pgm_inspect_v07.py` | Interactive PGM inspector + knob console; renders HTML + mermaid sidecar. |
| `tests/` | pytest suite covering curve identities, JAX≡numpy, joint prior behavior, fit pipeline. |

### Shared infrastructure (older but unchanged)

| File | What |
|---|---|
| `model.py` | Static (no learning) segment model: clean_dist, log_lik_survived/died, m_clear. Used by V07 numpy code via composition. |
| `hazard.py` | Hazard model interface and impls (haz1 piecewise, beta mixture). V07 hardcodes haz1. |
| `inference.py` | numpy MAP/Laplace helpers. Used by inspector and old experiments. |
| `config.py` | Cross-cutting constants (RESPAWN_MS, QUAD_POINTS, etc.). |

### Data layer

| File | What |
|---|---|
| `data/schema.sql` | Minimal sqlite table (`attempts`). |
| `data/db.py` | `init`, `export` CLI. Insert via your own pipeline. |

### Docs

| File | What |
|---|---|
| `external_docs/learning_curves_session_findings.md` | Plan-level (V07, DATA, HYPER, NUTS, JAX) — strategic decisions. |
| `external_docs/pgm_tricks.md` | Glossary of PGM patterns: identifiability, ridges, joint priors, partial pooling, Neal funnel, empirical Bayes. |
| `external_docs/segment_model.md`, `learning_curves.md` | Original model spec (numpy era). Still mostly correct. |

### Experiments / scratch

| Dir | What |
|---|---|
| `tmp/` | Throwaway scripts: smoke tests, benchmarks, ridge demos. Treat as a scratch dir — anything important gets promoted to `tests/` or `external_docs/`. |

## Public API surface

`import segments_v07 as sv` exposes:

```python
# Indices into the length-10 theta vector
sv.IDX_LOG_BPT, sv.IDX_LOG_SF_INF, sv.IDX_LOG_SF_1, sv.IDX_LOG_HALFLIFE_SF, ...

# Pure-math evals (JAX)
sv.learning_curve_at_n(theta, n)
sv.log_prior(theta)
sv.log_likelihood(theta, time_ms, is_died)
sv.log_posterior(theta, time_ms, is_died)
sv.halflife_sigma(A)
sv.initial_theta()

# Data wrangling
sv.data_to_arrays(attempts)        # list of (outcome, time_ms) -> jax arrays
sv.bucket_size(n)
sv.pad_to(time_ms, is_died, n_max)

# Fit
sv.warm_init_from_data(time_ms, is_died) -> theta_init
sv.prewarm_buckets()                                  # one-time JIT compile
sv.find_map(time_ms, is_died, theta_init=...,
            halflife_sf_pop_mean=..., halflife_sf_pop_sigma=...) -> (theta_map, info)
sv.laplace_posterior(theta_map, time_ms, is_died) -> (cov, ok)
sv.sample_laplace(theta_map, cov, n_samples=2000) -> samples

# Multi-segment empirical-Bayes pool on halflife_sf (HYPER-lite)
sv.fit_independent(segments) -> (maps, infos)         # baseline
sv.fit_eb_pool(segments, max_iter=10) -> (pool, maps, history)
# pool is sv.Pool(log_halflife_sf_mean, log_halflife_sf_sigma)

# Numpy reference (offline tools)
sv.numpy_reference.log_prior, .log_posterior, .sample_prior, ...
```

## Performance

On CPU JAX (after one-time `prewarm_buckets()`), single-segment at N=500:

| Op | Time |
|---|---|
| Cold MAP fit (with `warm_init_from_data`) | ~10–300 ms |
| Streaming refit (one new attempt, warm-started) | **p50 8ms, p99 35ms** |
| Laplace posterior (Hessian + invert + sample 2000) | ~600 ms |

Reference: numpy + Nelder-Mead takes ~70s for the same cold fit. JAX is ~700× faster.

GPU JAX (`pip install jax[cuda12]`) is the natural next step for batched
multi-segment fits via `jax.vmap`; see `learning_curves_session_findings.md`
Plan JAX.

## Known limitations

1. **MAP is a point estimate.** Use `laplace_posterior` for CIs in well-identified
   regimes. In the ridge case (regime 1, no learning — see `pgm_tricks.md`),
   Laplace's Hessian goes non-PD and bands are approximate. NUTS fallback is
   on the roadmap.
2. **Single-segment scale.** No hierarchical pooling yet (Plan HYPER). Each
   segment fit is independent.
3. **Hardcoded haz1.** The JAX path bakes in K=1 piecewise constant hazard.
   Other hazards (BetaMixtureHazard) only work via the numpy reference.
4. **CPU-only.** GPU bake-in is a follow-on.

## Development

```bash
# Run tests
.venv/Scripts/python -m pytest tests/ -v

# Run a benchmark
.venv/Scripts/python tmp/bench_full_streaming.py

# Run the recovery experiment (numpy, slow ~5 min)
.venv/Scripts/python phase2_recovery_v07.py --ns 500 --seeds 3

# Re-render inspector after a model change
.venv/Scripts/python pgm_inspect_v07.py --fit-n 500 --seed 1 --out tmp/v07.html
```

When adding code: tests first, doc second. Prior choices and architectural
notes go in `external_docs/` so the next person doesn't have to reverse-engineer
them. Tricks-and-gotchas patterns go in `pgm_tricks.md`.
