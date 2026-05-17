"""Empirical-Bayes halflife_sf pooling: does it recover the population mean,
and does it shrink a regime-1 segment toward that mean?

These are the two behaviors that justify HYPER-lite (see pgm_tricks.md and
external_docs/speed_and_techniques.md). If they don't hold, the EB code is
broken or the assumptions don't.
"""
import numpy as np
import pytest

import learning_model_v07 as lm
import fit_eb_pool as eb
from generate_synthetic_learning_v07 import (
    TRUTH_DEFAULTS, generate_multi_segment, generate,
)


@pytest.fixture(scope='module')
def pop_segments():
    """5 segments, halflife_sf truths drawn from Normal(log 50, 0.15) -- shifted
    from the prior median (log 28) so we can see the pool MOVE the prior.
    """
    return generate_multi_segment(
        n_segments=5, n_per_segment=300,
        true_pop_log_halflife_sf_mean=np.log(50.0),
        true_pop_log_halflife_sf_sigma=0.15,
        base_seed=1,
    )


def test_pool_recovers_population_mean(pop_segments):
    """After EB iteration, pool.mean should land near the true population
    log(50) ≈ 3.91, not the prior default log(28) ≈ 3.33.
    """
    pool, maps, history = eb.fit_eb_pool(pop_segments, max_iter=8)
    # Truth mean = log(50) ≈ 3.912
    assert 3.5 < pool.log_halflife_sf_mean < 4.2, (
        f"Pool mean {pool.log_halflife_sf_mean:.3f} (natural "
        f"{np.exp(pool.log_halflife_sf_mean):.1f}) is far from truth log(50)≈3.91"
    )
    # Sigma should be smallish (segments aren't wildly different)
    assert pool.log_halflife_sf_sigma < 0.5


def test_pool_converges_in_few_iters(pop_segments):
    """EB should reach tolerance within a handful of iterations."""
    pool, maps, history = eb.fit_eb_pool(pop_segments, max_iter=8, tol=1e-3)
    # Either converged before maxiter (history length <= maxiter) or final
    # delta tiny
    final_two = history[-2:]
    if len(final_two) == 2:
        delta_mean = abs(final_two[1].log_halflife_sf_mean
                         - final_two[0].log_halflife_sf_mean)
        assert delta_mean < 0.02, f"Final step still {delta_mean:.4f}"


def test_pool_shrinks_regime1_segment_toward_population_mean():
    """The killer use case: one regime-1 (no-learning) segment among many
    informative ones. Independent fit lands halflife near prior median (28).
    Pooled fit should pull it toward the population mean (~50).
    """
    # 4 informative segments with truth halflife=50, plus 1 regime-1 segment.
    informative = generate_multi_segment(
        n_segments=4, n_per_segment=300,
        true_pop_log_halflife_sf_mean=np.log(50.0),
        true_pop_log_halflife_sf_sigma=0.1,
        base_seed=2,
    )
    # Build the regime-1 segment manually: sf_1 = sf_inf (no sf learning).
    no_learn_truth = dict(TRUTH_DEFAULTS)
    no_learn_truth['sf_1'] = no_learn_truth['sf_inf']
    regime1 = {
        'data': generate(300, no_learn_truth, seed=999),
        'true_halflife_sf': float(no_learn_truth['sf_halflife']),
        'truth': no_learn_truth,
    }
    segments = informative + [regime1]

    # Baseline: independent fits
    indep_maps, _ = eb.fit_independent(segments)
    indep_hl_regime1 = float(np.exp(indep_maps[-1][lm.IDX_LOG_HALFLIFE_SF]))

    # Pooled fits
    pool, pooled_maps, _ = eb.fit_eb_pool(segments, max_iter=8)
    pooled_hl_regime1 = float(np.exp(pooled_maps[-1][lm.IDX_LOG_HALFLIFE_SF]))

    pop_target = float(np.exp(pool.log_halflife_sf_mean))

    # Independent regime-1 halflife should be near prior median (28).
    # Pooled regime-1 halflife should be near pop_target (~50).
    # The pooled value should be CLOSER to the pop_target than the independent
    # value is to the pop_target.
    indep_dist = abs(indep_hl_regime1 - pop_target)
    pooled_dist = abs(pooled_hl_regime1 - pop_target)
    assert pooled_dist < indep_dist, (
        f"Pool should pull regime-1 halflife toward population mean.\n"
        f"  population target: {pop_target:.1f}\n"
        f"  independent fit:   {indep_hl_regime1:.1f}  (distance {indep_dist:.1f})\n"
        f"  pooled fit:        {pooled_hl_regime1:.1f}  (distance {pooled_dist:.1f})"
    )


def test_pool_with_one_segment_equals_independent_fit():
    """Degenerate case: 1 segment means EB has no extra info; the pool just
    becomes that segment's MAP. Result should match (approximately) the
    independent fit."""
    seg = generate_multi_segment(
        n_segments=1, n_per_segment=300,
        true_pop_log_halflife_sf_mean=np.log(50.0),
        true_pop_log_halflife_sf_sigma=0.1,
        base_seed=3,
    )
    indep_maps, _ = eb.fit_independent(seg)
    pool, pooled_maps, _ = eb.fit_eb_pool(seg, max_iter=8)
    # Single-segment EB pulls toward itself; should converge to roughly
    # the independent estimate.
    indep_hl = float(np.exp(indep_maps[0][lm.IDX_LOG_HALFLIFE_SF]))
    pooled_hl = float(np.exp(pooled_maps[0][lm.IDX_LOG_HALFLIFE_SF]))
    rel_diff = abs(pooled_hl - indep_hl) / max(indep_hl, 1e-6)
    assert rel_diff < 0.30, (
        f"Single-segment pool diverged from independent fit: "
        f"indep={indep_hl:.1f}, pooled={pooled_hl:.1f}, rel_diff={rel_diff:.2f}"
    )
