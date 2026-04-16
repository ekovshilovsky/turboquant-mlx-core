#pragma once

#include <mlx/mlx.h>
#include "codebook.h"
#include "quantizer.h"

namespace turboquant {

/// Drop-in replacement for mlx nn::Linear that stores weights in
/// TurboQuant compressed format and uses fused dequant-matmul Metal kernels.
class TurboQuantLinear {
public:
    TurboQuantLinear(
        int in_features,
        int out_features,
        const QuantizedWeight& weights,
        const Codebook& primary_codebook,
        const Codebook& residual_codebook,
        uint32_t block_size);

    /// Forward pass: fused dequant + matmul in a single Metal dispatch.
    /// Input shape: [batch, in_features], Output shape: [batch, out_features].
    mlx::core::array forward(const mlx::core::array& input);

    int in_features() const;
    int out_features() const;

private:
    int in_features_;
    int out_features_;
    QuantizedWeight weights_;
    Codebook primary_codebook_;
    Codebook residual_codebook_;
    uint32_t block_size_;
};

} // namespace turboquant
