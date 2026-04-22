#pragma once

#include <mlx/mlx.h>
#include "codebook.h"
#include "quantizer.h"

namespace turboquant {

/// Drop-in replacement for mlx nn::Linear that stores weights in
/// TurboQuant compressed format and uses fused dequant-matmul Metal kernels.
///
/// Shard-aware construction: for a row-parallel tensor-parallel shard the
/// rank holds only a slice of the layer's input dimension. Pass the
/// original unsharded layer's in_features as full_in_features so the fused
/// kernel's scale factor matches the per-row norm correction baked during
/// offline quantisation. When full_in_features is zero (the default) the
/// kernel uses the rank-local in_features, which is correct for whole-
/// weight and column-parallel forwards.
class TurboQuantLinear {
public:
    TurboQuantLinear(
        int in_features,
        int out_features,
        const QuantizedWeight& weights,
        const Codebook& primary_codebook,
        const Codebook& residual_codebook,
        uint32_t block_size,
        int full_in_features = 0);

    /// Forward pass: fused dequant + matmul in a single Metal dispatch.
    /// Input shape: [batch, in_features], Output shape: [batch, out_features].
    mlx::core::array forward(const mlx::core::array& input);

    int in_features() const;
    int out_features() const;
    int full_in_features() const;

private:
    int in_features_;
    int out_features_;
    int full_in_features_;
    QuantizedWeight weights_;
    Codebook primary_codebook_;
    Codebook residual_codebook_;
    uint32_t block_size_;
};

} // namespace turboquant
