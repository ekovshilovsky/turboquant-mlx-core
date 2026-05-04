#pragma once

#include <mlx/mlx.h>
#include "codebook.h"
#include <cstdint>

namespace turboquant {

/// Configuration for the offline weight quantization pipeline.
struct QuantizerConfig {
    uint8_t primary_bits = 4;       ///< Bit width for primary pass
    uint8_t residual_bits = 4;      ///< Bit width for residual pass (0 = no residual)
    uint32_t block_size = 512;      ///< WHT block size for non-power-of-2 dims
    bool shared_rotation = false;   ///< When true, trades ~0.1% quality for 40% faster inference (single-WHT dequant)
    bool norm_correction = true;    ///< Apply corrected row norms
    int sensitive_layers_start = 0; ///< First N layers kept at fp16 (0 = none)
    int sensitive_layers_end = 0;   ///< Last N layers kept at fp16 (0 = none)
    /// Maximum tensor-parallel world size the produced snapshot is guaranteed
    /// to support. When > 1, per-layer block-size auto-selection adds the
    /// constraint (max_world_size * block_size) | in_features so row-parallel
    /// sharding in TurboQuantShardedToAllLinear can split each weight evenly
    /// across ranks. The default targets the common 2-Mac cluster topology;
    /// operators with specialized needs override explicitly:
    ///   * 1 — single-Mac max-quality snapshot, no tensor parallelism.
    ///   * 2 — default; reasonable per-layer quality, supports up to 2 ranks.
    ///   * 4+ — larger clusters; trades a small amount of per-layer quality
    ///     for the ability to shard across more ranks.
    uint32_t max_world_size = 2;
};

/// Result of quantizing a single weight matrix.
struct QuantizedWeight {
    mlx::core::array packed_primary;    ///< uint8 [out, in/2] for 4-bit or [out, in] for 5-bit primary
    mlx::core::array packed_residual;   ///< uint8 [out, in/2]: nibble-packed residual indices (3-4 bit)
    mlx::core::array norms;             ///< float16 [out]: corrected row norms
    mlx::core::array seeds;             ///< uint32 [3]: {seed_primary, seed_residual, block_size}
};

/// Apply Walsh-Hadamard rotation to a matrix using the given seed.
/// Operates on blocks of block_size for non-power-of-2 dimensions.
mlx::core::array apply_wht_rotation(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size);

/// Inverse Walsh-Hadamard rotation.
mlx::core::array apply_inverse_wht_rotation(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size);

/// Set thread-local flag to force CPU-only execution for all GPU-accelerated
/// operations (WHT rotation, Lloyd-Max codebook fitting). Must be called from
/// each worker thread in multi-threaded conversion contexts where concurrent
/// GPU Metal dispatch would cause command buffer conflicts. The flag is per-thread
/// so it does not affect other threads or single-threaded callers.
void set_force_cpu(bool force_cpu);

/// Query the thread-local CPU-only execution flag. Used internally by GPU-
/// accelerated codepaths to fall back to CPU when the calling thread has
/// indicated that GPU dispatch is unsafe (e.g., from a parallel converter worker).
bool get_force_cpu();

/// Quantize a single weight matrix through the full TurboQuant pipeline:
/// normalize -> WHT rotate -> Lloyd-Max quantize -> residual pass -> norm correction.
QuantizedWeight quantize_weight(
    const mlx::core::array& weight,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    const QuantizerConfig& config);

} // namespace turboquant
