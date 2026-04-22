#include "turboquant/linear.h"
#include "turboquant/dequantizer.h"

namespace turboquant {

TurboQuantLinear::TurboQuantLinear(
    int in_features,
    int out_features,
    const QuantizedWeight& weights,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size,
    int full_in_features)
    : in_features_(in_features)
    , out_features_(out_features)
    , full_in_features_(full_in_features > 0 ? full_in_features : in_features)
    , weights_(weights)
    , primary_codebook_(primary_codebook)
    , residual_codebook_(residual_codebook)
    , block_size_(block_size) {}

mlx::core::array TurboQuantLinear::forward(const mlx::core::array& input) {
    // Fused dequant-matmul: dequantize each WHT block on the fly and
    // accumulate dot products in a single Metal dispatch, avoiding the
    // intermediate full-precision weight materialization.
    //
    // full_in_features_ equals in_features_ for whole-weight and column-
    // parallel forwards, so the kernel's combined_scale is unchanged for
    // those callers. Row-parallel shards pass their original layer's
    // in_features and the kernel scales as if the full layer were present.
    return fused_dequant_matmul(
        weights_, primary_codebook_, residual_codebook_, block_size_, input,
        static_cast<uint32_t>(full_in_features_));
}

int TurboQuantLinear::in_features() const { return in_features_; }
int TurboQuantLinear::out_features() const { return out_features_; }
int TurboQuantLinear::full_in_features() const { return full_in_features_; }

} // namespace turboquant
