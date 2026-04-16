#pragma once

#include <mlx/mlx.h>
#include "codebook.h"
#include <cstdint>
#include <vector>

namespace turboquant {

/// Configuration for TurboQuant KV cache compression.
struct KVCacheConfig {
    int num_layers;             ///< Number of transformer layers
    int num_heads;              ///< Number of attention heads
    int head_dim;               ///< Dimension per head
    uint8_t k_bits = 3;        ///< K-cache quantization bits (+ 1-bit QJL residual)
    uint8_t v_bits = 3;        ///< V-cache quantization bits (PolarQuant only)
    int max_context = 1048576;  ///< Maximum context length
    int decode_window = 131072; ///< FP16 decode window size in tokens
};

/// Per-layer compressed KV storage. Packed 3-bit indices are stored 10 per
/// uint32 (30 bits used, 2 bits unused). Norms are stored as float per head
/// per token to preserve dynamic range during dequantization.
///
/// K-cache additionally stores 1-bit QJL residual signs and per-head QJL
/// norms to enable provably unbiased dot-product estimation (Zandieh et al.).
struct LayerKVStorage {
    std::vector<uint32_t> packed_key_indices;
    std::vector<uint32_t> packed_value_indices;
    std::vector<float> key_norms;     ///< [num_tokens * num_heads]
    std::vector<float> value_norms;   ///< [num_tokens * num_heads]
    std::vector<uint32_t> key_qjl_signs;  ///< 1-bit QJL residual signs, 32 per uint32
    std::vector<float> key_qjl_norms;     ///< QJL residual L2 norms [num_tokens * num_heads]
    int num_tokens = 0;
};

/// Manages TurboQuant-compressed KV cache with an incremental FP16 decode window.
/// Compressed storage holds the full context. The decode window holds recent
/// tokens in fp16 for fast SDPA attention. Tokens beyond the window are scored
/// using the fused TQ attention kernel.
class TQKVCache {
public:
    explicit TQKVCache(const KVCacheConfig& config);

    /// Append new key/value entries for a specific layer.
    /// Called per-chunk during prefill, per-token during decode.
    void append(int layer, const mlx::core::array& keys, const mlx::core::array& values);

    /// Get fp16 keys from the decode window for SDPA.
    mlx::core::array get_keys_fp16(int layer, int start, int end) const;

    /// Get fp16 values from the decode window for SDPA.
    mlx::core::array get_values_fp16(int layer, int start, int end) const;

    /// Run fused TQ attention on the compressed region (outside decode window).
    mlx::core::array compressed_attention(
        int layer,
        const mlx::core::array& queries,
        int compressed_start,
        int compressed_end) const;

    /// Current sequence length (total tokens stored).
    int seq_length() const;

    /// Reset cache state.
    void clear();

    const KVCacheConfig& config() const;

private:
    KVCacheConfig config_;
    Codebook codebook_3bit_;
    std::vector<LayerKVStorage> layers_;
    int seq_length_ = 0;

    /// Quantize a [num_tokens, num_heads, head_dim] tensor into packed 3-bit
    /// indices and per-head norms, appending the results to the layer storage.
    /// Used for V-cache (PolarQuant only, no QJL residual).
    void quantize_and_store(
        const mlx::core::array& data,
        uint32_t wht_seed,
        std::vector<uint32_t>& packed_indices,
        std::vector<float>& norms,
        int num_tokens);

    /// Quantize K-cache with 3-bit PolarQuant + 1-bit QJL residual correction.
    /// The QJL residual makes dot-product estimation provably unbiased, which
    /// is critical for K-cache since queries attend to keys via dot products.
    void quantize_and_store_with_qjl(
        const mlx::core::array& data,
        uint32_t wht_seed,
        uint32_t qjl_seed,
        LayerKVStorage& storage,
        int num_tokens);

    /// Compute 1-bit QJL residual signs and L2 norm for a single head vector.
    /// The residual is measured in the scaled+rotated domain after PolarQuant.
    void compute_qjl_residual(
        const float* rotated_scaled,
        const float* quantized_scaled,
        uint32_t qjl_seed,
        int head_dim,
        std::vector<uint32_t>& qjl_signs,
        std::vector<float>& qjl_norms);

    /// Reconstruct float32 vectors from packed 3-bit storage for a token range.
    mlx::core::array dequantize_range(
        const std::vector<uint32_t>& packed_indices,
        const std::vector<float>& norms,
        uint32_t wht_seed,
        int total_tokens,
        int start,
        int end) const;
};

} // namespace turboquant
