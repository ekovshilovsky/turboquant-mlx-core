#pragma once

#include <mlx/mlx.h>
#include "codebook.h"
#include "quantizer.h"

namespace turboquant {

/// Dequantize a weight matrix from packed TurboQuant representation.
/// CPU reference implementation for validation against Metal kernels.
mlx::core::array dequantize_weight_cpu(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size);

/// Dequantize using fused Metal kernel dispatch.
/// Dispatches turboquant_dequant.metal on GPU.
mlx::core::array dequantize_weight_gpu(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size);

/// Fused dequantization and matrix-vector multiplication in a single Metal
/// dispatch. Eliminates the intermediate full-precision weight matrix by
/// dequantizing each WHT block on the fly, multiplying by the corresponding
/// input activation, and accumulating partial dot products via parallel
/// reduction. Signs are computed from seeds using the same hash as
/// rotation.cpp::make_signs() — no buffer transfer needed.
///
/// Input shape:  [batch, in_features]  (any float type, cast to float16)
/// Output shape: [batch, out_features] (float16)
///
/// full_in_features: the original unsharded layer's in_features. When zero
///   (the default) the kernel uses the input tensor's in_features, which is
///   correct for whole-weight and column-parallel-sharded forwards. For
///   row-parallel shards pass the full layer's in_features so the kernel's
///   combined_scale reflects the pre-shard normalisation.
mlx::core::array fused_dequant_matmul(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size,
    const mlx::core::array& input,
    uint32_t full_in_features = 0);

} // namespace turboquant
