#include "turboquant/linear.h"
#include "turboquant/dequantizer.h"

namespace turboquant {

TurboQuantLinear::TurboQuantLinear(
    int in_features,
    int out_features,
    const QuantizedWeight& weights,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size)
    : in_features_(in_features)
    , out_features_(out_features)
    , weights_(weights)
    , primary_codebook_(primary_codebook)
    , residual_codebook_(residual_codebook)
    , block_size_(block_size) {}

mlx::core::array TurboQuantLinear::forward(const mlx::core::array& input) {
    // Fused dequant-matmul: dequantize each WHT block on the fly and
    // accumulate dot products in a single Metal dispatch, avoiding the
    // intermediate full-precision weight materialization.
    return fused_dequant_matmul(
        weights_, primary_codebook_, residual_codebook_, block_size_, input);
}

int TurboQuantLinear::in_features() const { return in_features_; }
int TurboQuantLinear::out_features() const { return out_features_; }

} // namespace turboquant
