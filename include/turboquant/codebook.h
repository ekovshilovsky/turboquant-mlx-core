#pragma once

#include <mlx/mlx.h>
#include <cstdint>
#include <vector>

namespace turboquant {

/// Where a codebook's centroids came from, which controls whether
/// strict symmetry (c[i] == -c[N-1-i]) is a required invariant.
/// Analytical codebooks are the precomputed optimum for N(0,1) and
/// are symmetric by construction. Empirical codebooks fit to real
/// data may legitimately capture distributional skew and need not
/// be symmetric.
enum class CodebookOrigin : uint8_t {
    Analytical,
    Empirical,
};

/// Lloyd-Max optimal centroids for quantizing coordinates after
/// Walsh-Hadamard rotation of normalized weight vectors.
/// Centroids are precomputed for the Beta distribution that arises
/// from WHT-rotated unit-norm vectors (Zandieh et al., ICLR 2026).
struct Codebook {
    std::vector<float> centroids;   ///< Sorted centroid values (2^bits entries)
    std::vector<float> boundaries;  ///< Decision boundaries between centroids
    uint8_t bits;                   ///< Quantization bit width (1-5)
    CodebookOrigin origin = CodebookOrigin::Analytical;
};

/// Generate Lloyd-Max codebook for the specified bit width.
/// Returns precomputed centroids optimal for the Beta distribution.
Codebook generate_codebook(uint8_t bits);

/// Fit a data-adaptive codebook via Lloyd-Max iteration on the supplied
/// samples. The resulting codebook is marked Empirical and is not
/// required to be symmetric around zero — real per-layer weight
/// distributions (e.g., post-WHT residual skew) often aren't, and
/// forcing symmetry would discard exactly the information per-layer
/// fitting is meant to capture. For the analytical symmetric N(0,1)
/// optimum, use generate_codebook(bits) instead.
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
