# Validation harness

Frozen TSV inputs + expected JSON payloads. Porters use these to verify
their re-implementation of the v1 contract reproduces this repo's
numerics.

```
validation/
├── inputs/
│   ├── n40_haz1.tsv         normal-N happy path (40 attempts, haz1 truth)
│   ├── n15_haz1.tsv         low-N case (15 attempts, exercises `low_n` caveat)
│   ├── no_learning.tsv      flat-curve truth — exercises (halflife, A) joint prior
│   ├── all_died.tsv         every attempt dies — exercises bpt-upper +inf fallback
│   ├── all_survived.tsv     no deaths — exercises all-survives likelihood path
│   └── pool3/
│       ├── seg-A.tsv        \
│       ├── seg-B.tsv         } 3-segment EB pool case
│       └── seg-C.tsv        /
├── expected/
│   ├── n40_haz1.json        ← fit_segment(attempts_n40, seed=0).
│   ├── n15_haz1.json        ← fit_segment(attempts_n15, seed=0).
│   ├── no_learning.json     ← flat-curve truth.
│   ├── all_died.json        ← honestly fires PPC tension (haz1 can't reproduce all-died).
│   ├── all_survived.json    ← alpha is prior-dominated.
│   └── pool3.json           ← fit_pool(segs, max_iter=3, seed=0).
└── regenerate.py
```

## For porters

Each case in `expected/` is the exact JSON payload produced by:

```python
import segments_v07 as sv
attempts = [{'outcome': ..., 'time_ms': ...}, ...]  # from inputs/<case>.tsv
result = sv.fit_segment(attempts, segment_id=<case>, n_samples=200, seed=0)
```

(`pool3.json` uses `sv.fit_pool(segs, n_samples=150, max_iter=3, seed=0)`.)

`wall_time_s` is stripped from the expected JSONs since it varies per run.

To validate your port, run the same inputs through your implementation
and deep-compare against the expected JSON with these tolerances (see
`external_docs/api_contract.md` "Stability promise" for the rationale):

| block | tolerance |
|---|---|
| envelope (`schema`, `kind`, `model`, `segment_id`, `n_attempts`, `caveats`) | exact |
| `status` (all five flags) | exact |
| `result.map`, `result.derived`, `result.pool` | relative 1e-5 (~5 sig figs) |
| `result.bands`, `result.ppc` | relative 1e-4 (~4 sig figs) |

The reference implementation of this comparison is
[`tests/test_validation_harness.py`](../tests/test_validation_harness.py)
— it's ~150 lines of standard library code, easy to port too.

## For this repo

The harness is run on every test invocation. If a benign refactor moves
numerics inside the tolerance window, no action needed. If a deliberate
change shifts numerics past tolerance:

```bash
python validation/regenerate.py
git diff validation/expected/
```

Inspect the diff. If the shift is expected, commit the updated JSONs
with a CHANGELOG note explaining the cause. If it's a regression, fix
the underlying change instead.

If you only changed code that the harness doesn't exercise (e.g. v2
files, beta2 internals), nothing in `expected/` should move. If the
harness still drifts after a no-op-looking change, suspect an
import-order shift; `regenerate.py` mirrors the test framework's
import order to keep JAX JIT compilation order stable.
