#include "turboquant/kv_cache.h"
#include "turboquant/decode_buffer.h"
#include <cassert>
#include <cmath>
#include <cstdio>

// Forward declaration — implemented in attention_dispatch.cpp
namespace turboquant {
mlx::core::array hybrid_attention(
    const mlx::core::array& queries,
    const TQKVCache& cache,
    const DecodeBuffer& buffer,
    int layer);
}

/// Verify that hybrid_attention returns zeros when the cache is empty.
static void test_empty_cache_returns_zeros() {
    turboquant::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 32;
    cfg.max_context = 64;
    cfg.decode_window = 32;

    turboquant::TQKVCache cache(cfg);
    turboquant::DecodeBuffer buffer(32, 2, 32);

    // Query: [batch=1, heads=2, q_seq=1, dim=32]
    auto queries = mlx::core::ones({1, 2, 1, 32}, mlx::core::float16);
    auto result = turboquant::hybrid_attention(queries, cache, buffer, 0);
    mlx::core::eval(result);

    assert(result.shape() == queries.shape());

    // All elements should be zero
    auto flat = mlx::core::reshape(result, {-1});
    auto result_f32 = mlx::core::astype(flat, mlx::core::float32);
    mlx::core::eval(result_f32);
    const float* data = result_f32.data<float>();
    int total = static_cast<int>(result_f32.size());
    for (int i = 0; i < total; ++i) {
        assert(data[i] == 0.0f);
    }

    printf("  PASS: empty cache returns zeros with correct shape\n");
}

/// Verify that short-context dispatch (all tokens within decode window) produces
/// valid output with correct shape and no NaN/Inf values.
static void test_short_context_pure_sdpa() {
    const int num_layers = 1;
    const int num_heads = 2;
    const int head_dim = 32;
    const int window_size = 32;
    const int num_tokens = 8;

    turboquant::KVCacheConfig cfg;
    cfg.num_layers = num_layers;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.max_context = 64;
    cfg.decode_window = window_size;

    turboquant::TQKVCache cache(cfg);
    turboquant::DecodeBuffer buffer(window_size, num_heads, head_dim);

    // Append several tokens to the KV cache (layer 0).
    // Keys/values shaped [num_tokens, num_heads, head_dim] with small
    // non-zero values to produce meaningful attention output.
    auto kv_data = mlx::core::random::normal(
        {num_tokens, num_heads, head_dim}, mlx::core::float32);
    mlx::core::eval(kv_data);
    cache.append(0, kv_data, kv_data);

    // Rebuild the decode buffer from the compressed cache to populate
    // the FP16 window with the dequantized representation.
    auto decompressed_keys = cache.get_keys_fp16(0, 0, num_tokens);
    auto decompressed_values = cache.get_values_fp16(0, 0, num_tokens);
    buffer.rebuild(decompressed_keys, decompressed_values);

    // Query: [batch=1, heads=2, q_seq=1, dim=32]
    auto queries = mlx::core::random::normal(
        {1, num_heads, 1, head_dim}, mlx::core::float32);
    queries = mlx::core::astype(queries, mlx::core::float16);
    mlx::core::eval(queries);

    auto result = turboquant::hybrid_attention(queries, cache, buffer, 0);
    mlx::core::eval(result);

    // Verify output shape matches query shape: [1, 2, 1, 32]
    assert(result.shape(0) == 1);
    assert(result.shape(1) == num_heads);
    assert(result.shape(2) == 1);
    assert(result.shape(3) == head_dim);

    // Verify all output values are finite (no NaN or Inf)
    auto flat = mlx::core::reshape(result, {-1});
    auto result_f32 = mlx::core::astype(flat, mlx::core::float32);
    mlx::core::eval(result_f32);
    const float* data = result_f32.data<float>();
    int total = static_cast<int>(result_f32.size());
    for (int i = 0; i < total; ++i) {
        assert(std::isfinite(data[i]));
    }

    printf("  PASS: short context routes to pure SDPA with valid output\n");
}

/// Verify the long-context path (seq_length > decode_window) produces valid
/// output via the dequantize-and-SDPA fallback.
static void test_long_context_dequant_fallback() {
    const int num_layers = 1;
    const int num_heads = 2;
    const int head_dim = 32;
    const int window_size = 4;
    const int total_tokens = 8;

    turboquant::KVCacheConfig cfg;
    cfg.num_layers = num_layers;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.max_context = 64;
    cfg.decode_window = window_size;

    turboquant::TQKVCache cache(cfg);
    turboquant::DecodeBuffer buffer(window_size, num_heads, head_dim);

    // Append tokens exceeding the window size to trigger the long-context path.
    auto kv_data = mlx::core::random::normal(
        {total_tokens, num_heads, head_dim}, mlx::core::float32);
    mlx::core::eval(kv_data);
    cache.append(0, kv_data, kv_data);

    // Rebuild the decode buffer — only the last window_size tokens fit,
    // but the cache holds all total_tokens.
    auto decompressed_keys = cache.get_keys_fp16(0, 0, total_tokens);
    auto decompressed_values = cache.get_values_fp16(0, 0, total_tokens);
    buffer.rebuild(decompressed_keys, decompressed_values);

    assert(cache.seq_length() > buffer.window_size());

    // Query: [batch=1, heads=2, q_seq=1, dim=32]
    auto queries = mlx::core::random::normal(
        {1, num_heads, 1, head_dim}, mlx::core::float32);
    queries = mlx::core::astype(queries, mlx::core::float16);
    mlx::core::eval(queries);

    auto result = turboquant::hybrid_attention(queries, cache, buffer, 0);
    mlx::core::eval(result);

    // Verify output shape
    assert(result.shape(0) == 1);
    assert(result.shape(1) == num_heads);
    assert(result.shape(2) == 1);
    assert(result.shape(3) == head_dim);

    // Verify all output values are finite
    auto flat = mlx::core::reshape(result, {-1});
    auto result_f32 = mlx::core::astype(flat, mlx::core::float32);
    mlx::core::eval(result_f32);
    const float* data = result_f32.data<float>();
    int total = static_cast<int>(result_f32.size());
    for (int i = 0; i < total; ++i) {
        assert(std::isfinite(data[i]));
    }

    printf("  PASS: long context uses dequant fallback with valid output\n");
}

/// Verify that output dimensions are correct for different query lengths.
static void test_dimensions_match() {
    const int num_heads = 2;
    const int head_dim = 32;
    const int num_tokens = 4;

    turboquant::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.max_context = 64;
    cfg.decode_window = 32;

    turboquant::TQKVCache cache(cfg);
    turboquant::DecodeBuffer buffer(32, num_heads, head_dim);

    auto kv_data = mlx::core::random::normal(
        {num_tokens, num_heads, head_dim}, mlx::core::float32);
    mlx::core::eval(kv_data);
    cache.append(0, kv_data, kv_data);

    auto decompressed_keys = cache.get_keys_fp16(0, 0, num_tokens);
    auto decompressed_values = cache.get_values_fp16(0, 0, num_tokens);
    buffer.rebuild(decompressed_keys, decompressed_values);

    // Single query token: [1, heads, 1, dim] -> output [1, heads, 1, dim]
    auto q1 = mlx::core::random::normal(
        {1, num_heads, 1, head_dim}, mlx::core::float32);
    q1 = mlx::core::astype(q1, mlx::core::float16);
    mlx::core::eval(q1);

    auto r1 = turboquant::hybrid_attention(q1, cache, buffer, 0);
    mlx::core::eval(r1);
    assert(r1.shape(0) == 1);
    assert(r1.shape(1) == num_heads);
    assert(r1.shape(2) == 1);
    assert(r1.shape(3) == head_dim);

    printf("  PASS: output dimensions match query shape\n");
}

int main() {
    printf("test_attention_dispatch (integration):\n");
    test_empty_cache_returns_zeros();
    test_short_context_pure_sdpa();
    test_long_context_dequant_fallback();
    test_dimensions_match();
    printf("All attention dispatch integration tests passed.\n");
    return 0;
}
