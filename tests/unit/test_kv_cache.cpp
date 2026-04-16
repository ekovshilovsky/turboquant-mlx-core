#include "turboquant/kv_cache.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>

using namespace turboquant;

/// Small configuration for fast unit tests: 2 layers, 2 heads, head_dim=64.
static KVCacheConfig test_config() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.head_dim = 64;
    cfg.k_bits = 3;
    cfg.v_bits = 3;
    cfg.max_context = 1024;
    cfg.decode_window = 256;
    return cfg;
}

/// Verify that appending a single token increments seq_length by 1.
static void test_append_increments_seq_length() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    assert(cache.seq_length() == 0);

    auto keys = mlx::core::random::normal({1, cfg.num_heads, cfg.head_dim});
    auto values = mlx::core::random::normal({1, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, values);

    cache.append(0, keys, values);
    assert(cache.seq_length() == 1);

    printf("  PASS: append single token increments seq_length\n");
}

/// Verify that appending multiple tokens at once increments seq_length by
/// the batch size.
static void test_append_batch_increments_seq_length() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    const int batch = 8;
    auto keys = mlx::core::random::normal({batch, cfg.num_heads, cfg.head_dim});
    auto values = mlx::core::random::normal({batch, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, values);

    cache.append(0, keys, values);
    assert(cache.seq_length() == batch);

    // Append a second batch
    auto keys2 = mlx::core::random::normal({4, cfg.num_heads, cfg.head_dim});
    auto values2 = mlx::core::random::normal({4, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys2, values2);

    cache.append(0, keys2, values2);
    assert(cache.seq_length() == batch + 4);

    printf("  PASS: append batch increments seq_length correctly\n");
}

/// Verify that clear() resets seq_length to 0.
static void test_clear_resets_state() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    auto keys = mlx::core::random::normal({5, cfg.num_heads, cfg.head_dim});
    auto values = mlx::core::random::normal({5, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, values);

    cache.append(0, keys, values);
    assert(cache.seq_length() == 5);

    cache.clear();
    assert(cache.seq_length() == 0);

    printf("  PASS: clear resets seq_length to zero\n");
}

/// Verify that get_keys_fp16 returns an array with the expected shape
/// [end - start, num_heads, head_dim].
static void test_retrieval_shape() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    const int num_tokens = 6;
    auto keys = mlx::core::random::normal({num_tokens, cfg.num_heads, cfg.head_dim});
    auto values = mlx::core::random::normal({num_tokens, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, values);

    cache.append(0, keys, values);

    // Retrieve a sub-range [2, 5)
    auto k_out = cache.get_keys_fp16(0, 2, 5);
    mlx::core::eval(k_out);
    assert(k_out.shape(0) == 3);
    assert(k_out.shape(1) == cfg.num_heads);
    assert(k_out.shape(2) == cfg.head_dim);

    auto v_out = cache.get_values_fp16(0, 0, num_tokens);
    mlx::core::eval(v_out);
    assert(v_out.shape(0) == num_tokens);
    assert(v_out.shape(1) == cfg.num_heads);
    assert(v_out.shape(2) == cfg.head_dim);

    printf("  PASS: retrieved arrays have correct shape\n");
}

/// Append random KV vectors and verify that the reconstructed values
/// approximately match the originals within expected quantization error.
/// Reports separate quality metrics for K-cache (3-bit + QJL) and V-cache
/// (3-bit only), including both per-element RMSE and dot-product error.
static void test_round_trip_quality() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    const int num_tokens = 4;
    auto keys = mlx::core::random::normal({num_tokens, cfg.num_heads, cfg.head_dim});
    auto values = mlx::core::random::normal({num_tokens, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, values);

    cache.append(0, keys, values);

    auto k_recovered = cache.get_keys_fp16(0, 0, num_tokens);
    auto v_recovered = cache.get_values_fp16(0, 0, num_tokens);

    // Cast everything to float32 for comparison
    auto keys_f32 = mlx::core::astype(keys, mlx::core::float32);
    auto k_rec_f32 = mlx::core::astype(k_recovered, mlx::core::float32);
    auto values_f32 = mlx::core::astype(values, mlx::core::float32);
    auto v_rec_f32 = mlx::core::astype(v_recovered, mlx::core::float32);
    mlx::core::eval(keys_f32, k_rec_f32, values_f32, v_rec_f32);

    const float* k_orig = keys_f32.data<float>();
    const float* k_rec = k_rec_f32.data<float>();
    const float* v_orig = values_f32.data<float>();
    const float* v_rec = v_rec_f32.data<float>();

    const int total_elems = num_tokens * cfg.num_heads * cfg.head_dim;

    // Compute relative RMSE for keys
    double k_err_sq = 0.0, k_orig_sq = 0.0;
    for (int i = 0; i < total_elems; ++i) {
        double diff = static_cast<double>(k_orig[i]) - static_cast<double>(k_rec[i]);
        k_err_sq += diff * diff;
        k_orig_sq += static_cast<double>(k_orig[i]) * static_cast<double>(k_orig[i]);
    }
    double k_rel_rmse = std::sqrt(k_err_sq / k_orig_sq);

    // Compute relative RMSE for values
    double v_err_sq = 0.0, v_orig_sq = 0.0;
    for (int i = 0; i < total_elems; ++i) {
        double diff = static_cast<double>(v_orig[i]) - static_cast<double>(v_rec[i]);
        v_err_sq += diff * diff;
        v_orig_sq += static_cast<double>(v_orig[i]) * static_cast<double>(v_orig[i]);
    }
    double v_rel_rmse = std::sqrt(v_err_sq / v_orig_sq);

    printf("  INFO: K-cache per-element relative RMSE = %.4f (3-bit PolarQuant + QJL)\n",
           k_rel_rmse);
    printf("  INFO: V-cache per-element relative RMSE = %.4f (3-bit PolarQuant only)\n",
           v_rel_rmse);

    // K-cache RMSE is based on 3-bit PolarQuant reconstruction only (QJL
    // correction improves dot-product estimation, not per-element accuracy)
    assert(k_rel_rmse < 0.25 && "K-cache round-trip relative RMSE exceeded 25%");
    assert(v_rel_rmse < 0.25 && "V-cache round-trip relative RMSE exceeded 25%");

    printf("  PASS: round-trip quality within expected quantization error bounds\n");
}

/// Verify that appending to different layers independently does not cause
/// cross-contamination between layer storage.
static void test_multi_layer_independence() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    // Append 3 tokens to layer 0, 7 tokens to layer 1
    auto keys0 = mlx::core::random::normal({3, cfg.num_heads, cfg.head_dim});
    auto vals0 = mlx::core::random::normal({3, cfg.num_heads, cfg.head_dim});
    auto keys1 = mlx::core::random::normal({7, cfg.num_heads, cfg.head_dim});
    auto vals1 = mlx::core::random::normal({7, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys0, vals0, keys1, vals1);

    cache.append(0, keys0, vals0);
    cache.append(1, keys1, vals1);

    // Global seq_length should be max(3, 7) = 7
    assert(cache.seq_length() == 7);

    // Layer 0 should have 3 tokens retrievable
    auto k0 = cache.get_keys_fp16(0, 0, 3);
    mlx::core::eval(k0);
    assert(k0.shape(0) == 3);

    // Layer 1 should have 7 tokens retrievable
    auto k1 = cache.get_keys_fp16(1, 0, 7);
    mlx::core::eval(k1);
    assert(k1.shape(0) == 7);

    // Retrieving beyond stored range on layer 0 should throw
    bool threw = false;
    try {
        auto bad = cache.get_keys_fp16(0, 0, 5);
        mlx::core::eval(bad);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw && "Expected out_of_range for over-range retrieval on layer 0");

    printf("  PASS: multi-layer storage is independent\n");
}

/// Verify that clear fully resets all layers and allows subsequent re-use.
static void test_clear_and_reuse() {
    auto cfg = test_config();
    TQKVCache cache(cfg);

    auto keys = mlx::core::random::normal({4, cfg.num_heads, cfg.head_dim});
    auto vals = mlx::core::random::normal({4, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys, vals);

    cache.append(0, keys, vals);
    cache.append(1, keys, vals);
    assert(cache.seq_length() == 4);

    cache.clear();
    assert(cache.seq_length() == 0);

    // Re-append after clear should work normally
    auto keys2 = mlx::core::random::normal({2, cfg.num_heads, cfg.head_dim});
    auto vals2 = mlx::core::random::normal({2, cfg.num_heads, cfg.head_dim});
    mlx::core::eval(keys2, vals2);

    cache.append(0, keys2, vals2);
    assert(cache.seq_length() == 2);

    auto k_out = cache.get_keys_fp16(0, 0, 2);
    mlx::core::eval(k_out);
    assert(k_out.shape(0) == 2);

    printf("  PASS: clear and reuse works correctly\n");
}

/// Verify that the QJL residual correction produces unbiased dot-product
/// estimates. Over many random Q/K pairs, the mean dot-product error should
/// converge toward zero. This is the fundamental property guaranteed by the
/// QJL estimator from Zandieh et al.
static void test_qjl_dot_product_fidelity() {
    auto cfg = test_config();

    // Run multiple independent trials to measure bias and variance of the
    // dot-product estimator with and without QJL correction.
    const int num_trials = 50;
    const int num_heads = cfg.num_heads;
    const int head_dim = cfg.head_dim;

    double sum_dot_error_k = 0.0;
    double sum_dot_error_v = 0.0;
    double sum_abs_dot_error_k = 0.0;
    double sum_abs_dot_error_v = 0.0;
    double sum_true_dot_sq = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        TQKVCache cache(cfg);

        // Generate random Q and K vectors (single token, all heads)
        auto q_arr = mlx::core::random::normal({1, num_heads, head_dim});
        auto k_arr = mlx::core::random::normal({1, num_heads, head_dim});
        auto v_arr = mlx::core::random::normal({1, num_heads, head_dim});
        mlx::core::eval(q_arr, k_arr, v_arr);

        // Store K and V in cache (K gets QJL, V does not)
        cache.append(0, k_arr, v_arr);

        // Retrieve reconstructed K and V
        auto k_rec = cache.get_keys_fp16(0, 0, 1);
        auto v_rec = cache.get_values_fp16(0, 0, 1);

        auto q_f32 = mlx::core::astype(q_arr, mlx::core::float32);
        auto k_f32 = mlx::core::astype(k_arr, mlx::core::float32);
        auto k_rec_f32 = mlx::core::astype(k_rec, mlx::core::float32);
        auto v_f32 = mlx::core::astype(v_arr, mlx::core::float32);
        auto v_rec_f32 = mlx::core::astype(v_rec, mlx::core::float32);
        mlx::core::eval(q_f32, k_f32, k_rec_f32, v_f32, v_rec_f32);

        const float* q_ptr = q_f32.data<float>();
        const float* k_ptr = k_f32.data<float>();
        const float* k_rec_ptr = k_rec_f32.data<float>();
        const float* v_ptr = v_f32.data<float>();
        const float* v_rec_ptr = v_rec_f32.data<float>();

        // Compute per-head dot products and accumulate errors
        for (int h = 0; h < num_heads; ++h) {
            int offset = h * head_dim;
            double true_dot_k = 0.0, est_dot_k = 0.0;
            double true_dot_v = 0.0, est_dot_v = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                true_dot_k += static_cast<double>(q_ptr[offset + d])
                              * static_cast<double>(k_ptr[offset + d]);
                est_dot_k += static_cast<double>(q_ptr[offset + d])
                             * static_cast<double>(k_rec_ptr[offset + d]);
                true_dot_v += static_cast<double>(q_ptr[offset + d])
                              * static_cast<double>(v_ptr[offset + d]);
                est_dot_v += static_cast<double>(q_ptr[offset + d])
                             * static_cast<double>(v_rec_ptr[offset + d]);
            }

            double err_k = est_dot_k - true_dot_k;
            double err_v = est_dot_v - true_dot_v;

            sum_dot_error_k += err_k;
            sum_dot_error_v += err_v;
            sum_abs_dot_error_k += std::abs(err_k);
            sum_abs_dot_error_v += std::abs(err_v);
            sum_true_dot_sq += true_dot_k * true_dot_k;
        }
    }

    int total_samples = num_trials * num_heads;
    double mean_error_k = sum_dot_error_k / total_samples;
    double mean_error_v = sum_dot_error_v / total_samples;
    double mean_abs_error_k = sum_abs_dot_error_k / total_samples;
    double mean_abs_error_v = sum_abs_dot_error_v / total_samples;
    double rms_true_dot = std::sqrt(sum_true_dot_sq / total_samples);

    printf("  INFO: K-cache dot-product mean bias  = %+.6f (should approach 0)\n",
           mean_error_k);
    printf("  INFO: V-cache dot-product mean bias  = %+.6f\n", mean_error_v);
    printf("  INFO: K-cache dot-product mean |err| = %.6f\n", mean_abs_error_k);
    printf("  INFO: V-cache dot-product mean |err| = %.6f\n", mean_abs_error_v);
    printf("  INFO: RMS true dot-product magnitude = %.4f\n", rms_true_dot);

    // The K-cache dot-product estimator should have low bias relative to the
    // magnitude of the true dot products. With QJL correction, the estimator
    // is provably unbiased in expectation.
    double relative_bias_k = std::abs(mean_error_k) / rms_true_dot;
    printf("  INFO: K-cache relative bias = %.4f (|mean_err| / rms_true_dot)\n",
           relative_bias_k);

    // Over 50 trials * 2 heads = 100 samples, the relative bias should be
    // small. We use a generous threshold since the PolarQuant base already
    // has low systematic error for random data.
    assert(relative_bias_k < 0.15 &&
           "K-cache dot-product estimator shows excessive bias");

    printf("  PASS: QJL dot-product fidelity verified (low bias across %d samples)\n",
           total_samples);
}

int main() {
    printf("test_kv_cache:\n");
    test_append_increments_seq_length();
    test_append_batch_increments_seq_length();
    test_clear_resets_state();
    test_retrieval_shape();
    test_round_trip_quality();
    test_qjl_dot_product_fidelity();
    test_multi_layer_independence();
    test_clear_and_reuse();
    printf("All KV cache tests passed.\n");
    return 0;
}
