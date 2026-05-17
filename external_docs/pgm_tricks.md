# PGM tricks & gotchas

Short glossary of patterns that come up when building probabilistic graphical
models. Captured from the segment-model work; meant to live as a checklist
when designing or debugging future PGMs. Each entry: name, when it bites,
what to do.

## Identifiability spectrum

Three flavors, in increasing severity:

- **Identifiable.** Data uniquely pins the parameter down in the
  infinite-data limit. Standard case.
- **Weakly identifiable.** Technically identifiable, but finite-noisy data
  leaves a wide elongated posterior. MAP is reproducible only loosely
  across seeds. Manifests as **ridges** in posterior geometry (see below).
- **Non-identifiable.** Data carries zero information about the parameter
  in some region of the other params' space. Likelihood is flat along the
  parameter's axis. Posterior = prior in that region.

Common cause: parameters that *only matter when another parameter is in
a particular range*. e.g. in our learning curves, `halflife` is
non-identifiable when `A = log(θ_1) - log(θ_∞) = 0` (no learning to
characterize).

## Ridges

A 1-D (or higher) direction in parameter space along which the likelihood
is nearly flat. Typically appears when two parameters trade off: doubling
one and halving the other gives the same likelihood.

**Symptoms:**
- MAP wanders across seeds at fixed truth.
- Asymptotic CIs are wrong (Hessian non-positive-definite at MAP).
- Posterior is a long thin sausage rather than a ball.

**Detection:** compute log_posterior on a 2-D slice through MAP for the
suspected pair. Plot as a heatmap; look for an elongated high-density
region. `tmp/ridge_demo.py` is an example.

**What fixes it (in order of complexity):**
1. Reparameterize to put the ridge in a more benign direction (sometimes
   possible, often not — algebraic equivalence preserves ridges).
2. Tighten the prior on one of the two params.
3. **Joint prior** (next entry).
4. More data of the right kind.
5. **Partial pooling** across many groups (hierarchical model).

## Joint priors

A.k.a. **dependent priors**: instead of factorizing the prior as
`P(A, B) = P(A) · P(B)`, model the joint directly so B's prior depends
on A.

**When to use:** when B is meaningless or non-identifiable for some
values of A, and you'd rather report a confident default than a wandering
estimate.

**Example (this project, `learning_model_v07.py`):** when `A=0` (no
learning), `halflife` is non-identifiable from data. We make
`σ_halflife(A) = σ_min + (σ_max - σ_min) · tanh(|A|)`. At `A=0`, σ shrinks
toward σ_min — prior dominates posterior; MAP returns prior median
confidently. At `|A|` large, σ expands toward σ_max — data dominates.

**The tanh smoothing matters.** A discontinuous switch (σ = σ_min if
A < ε, σ_max otherwise) breaks gradient-based optimizers and creates
posterior cliffs. tanh is smooth, monotonic, and asymptotically saturates.

**This is a meta-modeling choice, not structural.** You're not claiming
A and B are causally linked. You're encoding "report a default when the
question is meaningless." Worth a code comment to that effect.

**Compare to spike-and-slab:** `P(A) = π · δ(0) + (1-π) · Normal(...)`
is the cleanest treatment of "A might be exactly zero," but the point
mass breaks NM/L-BFGS and needs careful MCMC. Joint prior is the smooth
alternative.

## Hierarchical models / partial pooling / shrinkage

When you have many groups (segments, users, units), let each group's
params draw from a shared population:

```
For each group g:
    θ_g  ~  Normal(μ_pop, σ_pop)
μ_pop, σ_pop  ~  hyperpriors
```

**Effect (shrinkage):** each group's posterior on θ_g pulls toward μ_pop;
pull strength depends on how informative the group's own data is. Silent
groups shrink fully to μ_pop; data-rich groups barely move.

**When this is the answer:** when single-group identifiability is poor
(weak or non-identifiable) but you have many groups and at least some are
informative. The informative groups pin down (μ_pop, σ_pop) and that
inferred population prior fills in for the silent groups.

**What it doesn't do:** fix a structurally ill-posed single-group
likelihood. If literally one segment is non-identifiable on halflife AND
you have only one segment, you need a joint prior, not HYPER.

## Neal's funnel

The posterior geometry of centered hierarchical models pathologizes when
σ_pop is small (groups cluster tightly around μ_pop). The posterior in
(θ_g, σ_pop) space pinches into a funnel: the cross-section in θ_g
narrows as σ_pop → 0. HMC/NUTS samplers get stuck near the tip, taking
tiny steps and never exploring.

**Fix: non-centered reparameterization.** Replace
`θ_g ~ Normal(μ_pop, σ_pop)` with

```
z_g ~ Normal(0, 1)
θ_g = μ_pop + σ_pop · z_g     (deterministic)
```

The latent is now `z_g`, which is unit-variance regardless of σ_pop. The
funnel geometry vanishes; samplers explore freely.

Standard Stan/PyMC pattern. Mandatory for hierarchical MCMC at any
non-trivial dimension.

## Empirical Bayes & cached hyperparameters (scaling pattern)

Full hierarchical inference at the scale of thousands of groups is rarely
done jointly. Standard pattern:

1. Fit hyperparameters (μ_pop, σ_pop, ...) once using all data.
2. **Freeze them as point estimates.**
3. Per-group fits become independent and fully parallel — each is a
   small fit (10ish latents) conditional on frozen hypers.
4. Refit hypers periodically as new data accumulates.

This makes the PGM **decompose** into one shared hyperparameter block plus
N independent per-group blocks. Linear scaling. Production friendly.

**Tradeoff:** point-estimated hypers don't propagate hyperparameter
uncertainty into the per-group posteriors. For small N this matters; for
large N it's negligible.

## Other gotchas

### Optimizer stuck-at-init ≠ model bug

If MAP returns ≈ the initial point, check whether the *gradient at init*
is zero on the axes that didn't move. Common cause: initialization at a
point where some params are unidentifiable (e.g. flat-curve init makes
timescale params have zero gradient). Fix the init, not the model.

### MAP hides ridges

A single MAP point gives no indication of posterior elongation. Always
either:
- compute Hessian and inspect its condition number / eigenvalues, OR
- run multiple seeds and look at the spread of MAPs, OR
- compute log_posterior on a 2-D slice through MAP.

If asymptotic Hessian is non-PD, you have a ridge (or a saddle).

### Independent priors aren't always right

The default factorization `P(θ) = ∏ P(θ_i)` is convenient but isn't a
neutral choice — it's a strong claim that all the params are a priori
unrelated. Sometimes (often, in PGMs with weak identification) a joint
prior encodes your actual beliefs better.

### Reparameterizations preserve ridges

If `(A, B)` are ridged, then any smooth invertible reparameterization
`(A', B') = f(A, B)` is ridged in the new coords (Jacobian moves the
geometry but doesn't flatten it). Renaming doesn't fix identifiability.
Only adding information does — via data, prior, or pooling.

### "Sub-PGM decomposition" = empirical Bayes

When the joint PGM is too big to fit, factor it into blocks: a small
shared "hyperparameter" block plus many independent per-group blocks. Fit
the hyperparameter block once and freeze it. This is the standard pattern
for production hierarchical models at scale; not a hack.

## Cross-references in this codebase

- Joint prior implementation: `learning_model_v07.py` `halflife_sigma()`
  and the comment block above it.
- Ridge demonstrations: `tmp/ridge_demo.py` (strong learning) and
  `tmp/ridge_demo_no_learning.py` (regime-1 no-learning).
- Plan-level commitments: `reports/2026-05-16_learning_curves_findings.md`
  Plans HYPER (partial pooling), NUTS (sampler), JAX (autodiff).
