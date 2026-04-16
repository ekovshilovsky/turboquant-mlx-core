#include "turboquant/kv_cache.h"
#include "turboquant/decode_buffer.h"

#include <cmath>

namespace turboquant {

/// Hybrid attention dispatch: routes to SDPA for recent tokens (decode window)
/// and to fused TQ attention kernel for compressed distant tokens.
///
/// Strategy:
///   - Short context (seq_length <= decode_window): All tokens reside in the
///     FP16 decode buffer. Route entirely through MLX native SDPA — no
///     dequantization overhead, no compressed-path latency.
///   - Long context (seq_length > decode_window): Fall back to full
///     dequantization of the compressed KV range and run SDPA over the entire
///     context. The fused TQ attention Metal kernel is compiled and available,
///     but wiring its dispatch through the MLX metal_kernel C++ API requires
///     matching the complex parameter layout; this will be connected in the
///     optimization pass to avoid the dequantization cost on long contexts.
mlx::core::array hybrid_attention(
    const mlx::core::array& queries,
    const TQKVCache& cache,
    const DecodeBuffer& buffer,
    int layer) {

    const int seq_len = cache.seq_length();

    // No KV entries to attend to — return zeros matching query shape.
    if (seq_len == 0) {
        return mlx::core::zeros_like(queries);
    }

    const int head_dim = cache.config().head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (seq_len <= buffer.window_size()) {
        // All tokens fit within the decode window — use native MLX SDPA
        // on the FP16 buffer contents directly, avoiding dequantization.
        auto K = buffer.keys(0, seq_len);    // [seq, heads, dim]
        auto V = buffer.values(0, seq_len);  // [seq, heads, dim]

        // MLX SDPA expects [batch, heads, seq, dim]. The decode buffer
        // returns [seq, heads, dim], so transpose to [heads, seq, dim]
        // then add a leading batch dimension of 1.
        K = mlx::core::transpose(K, {1, 0, 2});  // [heads, seq, dim]
        V = mlx::core::transpose(V, {1, 0, 2});  // [heads, seq, dim]
        K = mlx::core::expand_dims(K, 0);         // [1, heads, seq, dim]
        V = mlx::core::expand_dims(V, 0);         // [1, heads, seq, dim]

        // Ensure queries are in the same dtype as the KV tensors (float16)
        auto Q = mlx::core::astype(queries, K.dtype());

        return mlx::core::fast::scaled_dot_product_attention(
            Q, K, V, scale);
    }

    // Long context: dequantize the full compressed range and run SDPA.
    // This path materialises all KV as FP16, which is correct but not
    // memory-optimal. The fused compressed-attention kernel will replace
    // this once its C++ dispatch is wired.
    auto K = cache.get_keys_fp16(layer, 0, seq_len);    // [seq, heads, dim]
    auto V = cache.get_values_fp16(layer, 0, seq_len);  // [seq, heads, dim]

    K = mlx::core::transpose(K, {1, 0, 2});  // [heads, seq, dim]
    V = mlx::core::transpose(V, {1, 0, 2});  // [heads, seq, dim]
    K = mlx::core::expand_dims(K, 0);         // [1, heads, seq, dim]
    V = mlx::core::expand_dims(V, 0);         // [1, heads, seq, dim]

    auto Q = mlx::core::astype(queries, K.dtype());

    return mlx::core::fast::scaled_dot_product_attention(
        Q, K, V, scale);
}

} // namespace turboquant
