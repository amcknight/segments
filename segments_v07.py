"""Public API for the V07 segment model. Single import surface for downstream
Python consumers.

Example:
    import segments_v07 as sv

    # Optional: pre-warm the JAX cache at process start
    sv.prewarm_buckets()

    # Cold fit from data alone
    time_ms, is_died = sv.data_to_arrays(attempts)
    theta_init = sv.warm_init_from_data(time_ms, is_died)
    theta_map, info = sv.find_map(time_ms, is_died, theta_init=theta_init)

    # Posterior summary
    cov, ok = sv.laplace_posterior(theta_map, time_ms, is_died)
    samples = sv.sample_laplace(theta_map, cov, n_samples=2000)

    # Streaming: next attempt comes in, warm-start from previous MAP
    time_ms2, is_died2 = sv.data_to_arrays(attempts_plus_one)
    theta_map2, info2 = sv.find_map(time_ms2, is_died2, theta_init=theta_map)

Theta layout (length 10, log-space) is constants below. Curve evaluation:
    log_theta(n) = log_theta_inf + (log_theta_1 - log_theta_inf) * 2^(-(n-1)/halflife)

Everything imported here is verified by the test suite in tests/.
"""

# ----- Model constants -----
from learning_model_v07 import (
    N_PARAMS,
    IDX_LOG_BPT,
    IDX_LOG_SF_INF, IDX_LOG_SSP_INF, IDX_LOG_ALPHA_INF,
    IDX_LOG_SF_1,   IDX_LOG_SSP_1,   IDX_LOG_ALPHA_1,
    IDX_LOG_HALFLIFE_SF, IDX_LOG_HALFLIFE_SSP, IDX_LOG_HALFLIFE_ALPHA,
)

# ----- Pure-math eval (JAX, for callers that need to compute curves) -----
from learning_model_v07_jax import (
    learning_curve_at_n,
    log_prior,
    log_likelihood,
    log_posterior,
    initial_theta,
    data_to_arrays,
    bucket_size,
    pad_to,
    halflife_sigma,
)

# Prior sampling lives in the numpy reference module (use numpy_reference.sample_prior).

# ----- Fit + posterior -----
from fit_jax import (
    warm_init_from_data,
    prewarm_buckets,
    find_map,
    laplace_posterior,
    sample_laplace,
)

# ----- Multi-segment empirical-Bayes pool (HYPER-lite, all 3 halflives) -----
from fit_eb_pool import (
    Pool,                # NamedTuple with 6 fields: log_halflife_{sf,ssp,alpha}_{mean,sigma}
    fit_eb_pool,         # one-step sigma + iterated mean, returns (pool, maps, history)
    fit_independent,    # baseline: no pooling
)

# ----- Numpy reference (kept for cross-validation and offline tools) -----
import learning_model_v07 as numpy_reference

__all__ = [
    # constants
    'N_PARAMS',
    'IDX_LOG_BPT',
    'IDX_LOG_SF_INF',  'IDX_LOG_SSP_INF',  'IDX_LOG_ALPHA_INF',
    'IDX_LOG_SF_1',    'IDX_LOG_SSP_1',    'IDX_LOG_ALPHA_1',
    'IDX_LOG_HALFLIFE_SF', 'IDX_LOG_HALFLIFE_SSP', 'IDX_LOG_HALFLIFE_ALPHA',
    # eval
    'learning_curve_at_n',
    'log_prior', 'log_likelihood', 'log_posterior',
    'initial_theta', 'data_to_arrays',
    'halflife_sigma',
    'bucket_size', 'pad_to',
    # fit
    'warm_init_from_data', 'prewarm_buckets', 'find_map',
    'laplace_posterior', 'sample_laplace',
    # EB pool
    'Pool', 'fit_eb_pool', 'fit_independent',
    # reference (numpy: sample_prior, log_prior, log_likelihood etc.)
    'numpy_reference',
]
