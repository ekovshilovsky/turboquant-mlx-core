#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <vector>

namespace turboquant {

/// Lloyd-Max optimal centroids for quantizing coordinates after
/// Walsh-Hadamard rotation of normalized weight vectors.
/// Centroids are precomputed for the Beta distribution that arises
/// from WHT-rotated unit-norm vectors (Zandieh et al., ICLR 2026).
struct Codebook {
    std::vector<float> centroids;   ///< Sorted centroid values (2^bits entries)
    std::vector<float> boundaries;  ///< Decision boundaries between centroids
    uint8_t bits;                   ///< Quantization bit width (1-5)
};

/// Generate Lloyd-Max codebook for the specified bit width.
/// Returns precomputed centroids optimal for the Beta distribution.
Codebook generate_codebook(uint8_t bits);

/// Generate a Lloyd-Max codebook refined on actual data values.
/// Starts from the precomputed N(0,1) centroids and iteratively adjusts
/// them to minimize mean squared quantization error for the empirical
/// distribution of the provided samples. Converges in 50-100 iterations
/// for typical weight distributions.
///
/// The fit is performed on |data| against the positive half of the
/// codebook, and the full codebook is assembled by mirroring. This
/// guarantees exact bit-level symmetry (c[i] == -c[N-1-i]) regardless
/// of finite-sample asymmetry in the input, at the cost of modeling the
/// distribution as symmetric around zero — appropriate for WHT-rotated
/// weights and other near-Gaussian inputs this library targets.
Codebook generate_codebook_from_data(const std::vector<float>& data, uint8_t bits, int iterations = 100);

/// Map continuous values to nearest centroid indices.
/// Input values are expected to be WHT-rotated and normalized.
mlx::core::array quantize(const mlx::core::array& values, const Codebook& codebook);

/// Map centroid indices back to centroid values.
mlx::core::array dequantize(const mlx::core::array& indices, const Codebook& codebook);

/// Validate codebook correctness: symmetry, sorting, boundary placement.
/// Returns true if all invariants hold.
bool validate_codebook(const Codebook& codebook);

} // namespace turboquant
