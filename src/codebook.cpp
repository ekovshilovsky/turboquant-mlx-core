#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace turboquant {

namespace {

/// Precomputed Lloyd-Max optimal centroids for N(0,1) at each supported bit width.
/// Source: Zandieh et al. (ICLR 2026) / TheTom/llama-cpp-turboquant (MIT).
static const float kCentroids1Bit[] = {
    -0.7978845608f, 0.7978845608f
};

static const float kCentroids2Bit[] = {
    -1.510232f, -0.4527800f, 0.4527800f, 1.510232f
};

static const float kCentroids3Bit[] = {
    -2.152176f, -1.344318f, -0.756130f, -0.245379f,
     0.245379f,  0.756130f,  1.344318f,  2.152176f
};

static const float kCentroids4Bit[] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

static const float kCentroids5Bit[] = {
    -3.255544f, -2.685239f, -2.311434f, -2.022174f,
    -1.780580f, -1.569614f, -1.379896f, -1.205655f,
    -1.043046f, -0.889350f, -0.742542f, -0.601049f,
    -0.463598f, -0.329119f, -0.196679f, -0.065429f,
     0.065429f,  0.196679f,  0.329119f,  0.463598f,
     0.601049f,  0.742542f,  0.889350f,  1.043046f,
     1.205655f,  1.379896f,  1.569614f,  1.780580f,
     2.022174f,  2.311434f,  2.685239f,  3.255544f
};

/// Build a Codebook from a raw centroid array, computing decision boundaries
/// as midpoints between adjacent centroids.
Codebook build_from_centroids(const float* data, size_t count, uint8_t bits) {
    std::vector<float> centroids(data, data + count);
    std::vector<float> boundaries;
    boundaries.reserve(count - 1);
    for (size_t i = 0; i + 1 < count; i++) {
        boundaries.push_back((centroids[i] + centroids[i + 1]) * 0.5f);
    }
    return Codebook{std::move(centroids), std::move(boundaries), bits};
}

// Minimum sample count to justify GPU dispatch for Lloyd-Max iteration.
// Below this threshold the CPU binary search loop is faster than MLX kernel
// launch overhead and GPU memory transfer latency.
constexpr size_t kLloydMaxGpuThreshold = 65536;

// Lloyd-Max iteration exits early when no centroid moves more than this
// per step. Tighter thresholds rarely improve the fit meaningfully and
// cost iterations; looser thresholds can stop short of the empirical
// optimum.
constexpr float kLloydMaxShiftThreshold = 1e-6f;

// Assemble a full symmetric codebook from a fitted positive half.
//
// Finite samples from a near-symmetric distribution are never perfectly
// symmetric, so fitting Lloyd-Max on the raw data produces centroids that
// inherit the empirical asymmetry of the input. The downstream invariant
// c[i] == -c[N-1-i] (enforced by validate_codebook and relied on by every
// consumer of the codebook) therefore cannot be achieved by convergence
// alone. Folding the input to |x| and fitting only the positive half, then
// mirroring it, enforces symmetry exactly by construction.
//
// The mapping places the fitted positive centroids in the upper half and
// their negated counterparts in the lower half, preserving strict sort
// order across the full codebook. The center boundary between the smallest
// positive and smallest negative centroid is exactly 0 by construction.
Codebook assemble_symmetric_codebook(
    const std::vector<float>& positive_centroids, uint8_t bits) {
    const size_t half = positive_centroids.size();
    const size_t num_centroids = half * 2;

    std::vector<float> centroids(num_centroids);
    for (size_t i = 0; i < half; ++i) {
        centroids[half + i]     =  positive_centroids[i];
        centroids[half - 1 - i] = -positive_centroids[i];
    }

    std::vector<float> boundaries(num_centroids - 1);
    for (size_t i = 0; i + 1 < num_centroids; ++i) {
        boundaries[i] = (centroids[i] + centroids[i + 1]) * 0.5f;
    }

    return Codebook{std::move(centroids), std::move(boundaries), bits};
}

// Extract the initialization for the positive-half Lloyd-Max fit from the
// precomputed analytical codebook for N(0,1). The analytical codebook is
// already symmetric, so its upper half is the correct starting point for
// fitting a positive-only distribution.
std::vector<float> initial_positive_centroids(uint8_t bits) {
    Codebook analytical = generate_codebook(bits);
    const size_t half = analytical.centroids.size() / 2;
    return std::vector<float>(
        analytical.centroids.begin() + static_cast<std::ptrdiff_t>(half),
        analytical.centroids.end());
}

} // namespace

Codebook generate_codebook(uint8_t bits) {
    switch (bits) {
        case 1: return build_from_centroids(kCentroids1Bit, 2,  bits);
        case 2: return build_from_centroids(kCentroids2Bit, 4,  bits);
        case 3: return build_from_centroids(kCentroids3Bit, 8,  bits);
        case 4: return build_from_centroids(kCentroids4Bit, 16, bits);
        case 5: return build_from_centroids(kCentroids5Bit, 32, bits);
        default:
            throw std::invalid_argument("Unsupported bit width; must be 1-5");
    }
}

/// CPU-only Lloyd-Max iteration for small datasets. Uses binary search over
/// boundaries for bin assignment, which is efficient for datasets below the
/// GPU dispatch threshold where kernel launch overhead would dominate.
///
/// The input is folded to |x| and only the positive half of the codebook is
/// fit; the full codebook is assembled by mirroring. This guarantees exact
/// symmetry (c[i] == -c[N-1-i]) without relying on convergence of an
/// empirically-asymmetric distribution.
///
/// External linkage is retained (no `static`) so the CPU and GPU paths can
/// be invoked directly from tests that verify both fits produce equivalent
/// codebooks on identical input. The public entry point is still
/// `generate_codebook_from_data`, which dispatches by sample count.
Codebook generate_codebook_from_data_cpu(
    const std::vector<float>& data, uint8_t bits, int iterations) {

    std::vector<float> positive = initial_positive_centroids(bits);
    const size_t half = positive.size();
    const size_t n = data.size();

    // Positive-half bin boundaries. The first bin covers [0, boundaries[0]),
    // interior bins [boundaries[i-1], boundaries[i]), and the last bin
    // [boundaries[half-2], +inf). This mirrors the layout used for the full
    // codebook but operates on |x| so no negative-bin book-keeping is needed.
    std::vector<float> pos_boundaries(half > 0 ? half - 1 : 0);

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 1 < half; ++i) {
            pos_boundaries[i] = (positive[i] + positive[i + 1]) * 0.5f;
        }

        std::vector<double> bin_sum(half, 0.0);
        std::vector<size_t> bin_count(half, 0);

        for (size_t i = 0; i < n; ++i) {
            float abs_v = std::abs(data[i]);
            auto it = std::lower_bound(pos_boundaries.begin(), pos_boundaries.end(), abs_v);
            size_t idx = static_cast<size_t>(it - pos_boundaries.begin());
            if (idx >= half) idx = half - 1;
            bin_sum[idx] += static_cast<double>(abs_v);
            bin_count[idx]++;
        }

        float max_shift = 0.0f;
        for (size_t i = 0; i < half; ++i) {
            if (bin_count[i] > 0) {
                float new_centroid = static_cast<float>(bin_sum[i] / static_cast<double>(bin_count[i]));
                float shift = std::abs(new_centroid - positive[i]);
                if (shift > max_shift) max_shift = shift;
                positive[i] = new_centroid;
            }
        }

        if (max_shift < kLloydMaxShiftThreshold) break;
    }

    return assemble_symmetric_codebook(positive, bits);
}

/// GPU-accelerated Lloyd-Max iteration using MLX array operations.
/// For large datasets (millions of weight values), the per-bin masked
/// sum and count operations run on GPU through MLX's lazy evaluation,
/// avoiding the serial binary search loop over every element.
///
/// The input is folded to |x| on the GPU and the fit is performed on the
/// positive half of the codebook only. The full codebook is assembled by
/// mirroring the fitted positive centroids. This enforces the symmetry
/// invariant c[i] == -c[N-1-i] exactly by construction rather than hoping
/// that convergence over an empirically-asymmetric finite sample happens
/// to produce it.
///
/// Each iteration constructs per-bin masks over |x| using boundary
/// comparisons, computes masked sums and counts via GPU reductions, and
/// updates positive centroids on the host. The boundary comparisons and
/// reductions are the dominant cost and parallelize well on the GPU ALUs.
///
/// External linkage is retained (no `static`) so the CPU and GPU paths can
/// be invoked directly from tests that verify both fits produce equivalent
/// codebooks on identical input. The public entry point is still
/// `generate_codebook_from_data`, which dispatches by sample count.
Codebook generate_codebook_from_data_gpu(
    const std::vector<float>& data, uint8_t bits, int iterations) {

    std::vector<float> positive = initial_positive_centroids(bits);
    const int half = static_cast<int>(positive.size());

    // Transfer the data to GPU once and fold to |x|; the folded array is
    // reused across all iterations so the reduction operates on the half
    // of the distribution the positive centroids actually represent.
    auto data_arr = mlx::core::array(
        data.data(),
        {static_cast<int>(data.size())},
        mlx::core::float32);
    auto abs_data = mlx::core::abs(data_arr);

    std::vector<float> pos_boundaries(half > 0 ? half - 1 : 0);

    for (int iter = 0; iter < iterations; ++iter) {
        // Recompute positive-half boundaries as midpoints between adjacent
        // positive centroids. Bin 0 covers [0, boundaries[0]), interior bins
        // cover [boundaries[i-1], boundaries[i]), and the last bin covers
        // [boundaries[half-2], +inf).
        for (int i = 0; i + 1 < half; ++i) {
            pos_boundaries[i] = (positive[i] + positive[i + 1]) * 0.5f;
        }

        // Build boundary array on GPU for vectorized bin assignment.
        // For typical bit widths (2-5 bits = 2-16 positive centroids), the
        // boundary array is tiny and the overhead is negligible compared to
        // the reduction over millions of data elements.
        auto boundaries_arr = mlx::core::array(
            pos_boundaries.data(),
            {static_cast<int>(pos_boundaries.size())},
            mlx::core::float32);

        // Compute per-bin masked sums and counts using GPU reductions.
        // Each bin is defined by its lower and upper boundaries. The
        // comparison chain produces a boolean mask per bin, which is
        // multiplied element-wise with |x| to produce masked values for
        // summation. This replaces a per-element binary search with
        // N/2 parallel GPU comparisons.
        //
        // MLX reductions run in float32 with implementation-defined summation
        // order (typically a tree reduction). The CPU path uses double-precision
        // accumulation for stability. Expect ~1-ULP drift between the two paths
        // at the centroid level — still well under float16 storage precision.
        std::vector<mlx::core::array> bin_sums;
        std::vector<mlx::core::array> bin_counts;
        bin_sums.reserve(static_cast<size_t>(half));
        bin_counts.reserve(static_cast<size_t>(half));

        for (int bin = 0; bin < half; ++bin) {
            mlx::core::array mask(0.0f);
            if (bin == 0 && bin == half - 1) {
                // Single-bin edge case (1-bit codebook): all elements belong
                // to the sole positive bin.
                mask = mlx::core::astype(
                    mlx::core::ones_like(abs_data), mlx::core::bool_);
            } else if (bin == 0) {
                // First positive bin: |x| below the first positive boundary
                auto hi = mlx::core::take(boundaries_arr, 0);
                mask = mlx::core::less(abs_data, hi);
            } else if (bin == half - 1) {
                // Last positive bin: |x| at or above the last positive boundary
                auto lo = mlx::core::take(boundaries_arr, bin - 1);
                mask = mlx::core::greater_equal(abs_data, lo);
            } else {
                // Interior positive bin: |x| in [boundaries[bin-1], boundaries[bin])
                auto lo = mlx::core::take(boundaries_arr, bin - 1);
                auto hi = mlx::core::take(boundaries_arr, bin);
                auto ge = mlx::core::greater_equal(abs_data, lo);
                auto lt = mlx::core::less(abs_data, hi);
                mask = mlx::core::logical_and(ge, lt);
            }

            auto mask_f = mlx::core::astype(mask, mlx::core::float32);
            auto masked_data = mlx::core::multiply(abs_data, mask_f);
            bin_sums.push_back(mlx::core::sum(masked_data));
            bin_counts.push_back(mlx::core::sum(mask_f));
        }

        // Materialize all bin reductions in a single GPU dispatch to maximize
        // occupancy and minimize synchronization points
        std::vector<mlx::core::array> to_materialize;
        to_materialize.reserve(static_cast<size_t>(half) * 2);
        for (int bin = 0; bin < half; ++bin) {
            to_materialize.push_back(bin_sums[bin]);
            to_materialize.push_back(bin_counts[bin]);
        }
        mlx::core::eval(to_materialize);

        // Update positive centroids from the GPU-computed sums and counts.
        // Empty bins retain their previous centroid to maintain coverage.
        float max_shift = 0.0f;
        for (int bin = 0; bin < half; ++bin) {
            float count = bin_counts[bin].item<float>();
            if (count > 0.0f) {
                float new_centroid = bin_sums[bin].item<float>() / count;
                float shift = std::abs(new_centroid - positive[bin]);
                if (shift > max_shift) max_shift = shift;
                positive[bin] = new_centroid;
            }
        }

        // Terminate early when all centroids have stabilized. WHT-rotated
        // weight distributions are near-Gaussian, so convergence from the
        // N(0,1) initial guess typically completes in 20-30 iterations.
        if (max_shift < kLloydMaxShiftThreshold) break;
    }

    return assemble_symmetric_codebook(positive, bits);
}

Codebook generate_codebook_from_data(const std::vector<float>& data, uint8_t bits, int iterations) {
    if (data.empty()) {
        return generate_codebook(bits);
    }

    // Route to GPU for large datasets where parallel reductions outperform
    // the serial binary search loop. The threshold is tuned for Apple Silicon
    // where GPU kernel launch overhead is approximately 50-100us. The GPU path
    // is disabled when the thread-local CPU-only flag is set (converter workers).
    if (data.size() >= kLloydMaxGpuThreshold && !get_force_cpu()) {
        return generate_codebook_from_data_gpu(data, bits, iterations);
    }
    return generate_codebook_from_data_cpu(data, bits, iterations);
}

mlx::core::array quantize(const mlx::core::array& values, const Codebook& codebook) {
    mlx::core::eval(const_cast<mlx::core::array&>(values));

    const float* in = values.data<float>();
    size_t n = static_cast<size_t>(values.size());
    const auto& bounds = codebook.boundaries;
    const size_t num_centroids = codebook.centroids.size();

    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; i++) {
        // Binary-search the decision boundaries to find the owning centroid bin
        auto it = std::lower_bound(bounds.begin(), bounds.end(), in[i]);
        size_t idx = static_cast<size_t>(it - bounds.begin());
        if (idx >= num_centroids) idx = num_centroids - 1;
        out[i] = static_cast<uint8_t>(idx);
    }

    return mlx::core::array(out.data(), {static_cast<int>(n)}, mlx::core::uint8);
}

mlx::core::array dequantize(const mlx::core::array& indices, const Codebook& codebook) {
    mlx::core::eval(const_cast<mlx::core::array&>(indices));

    const uint8_t* in = indices.data<uint8_t>();
    size_t n = static_cast<size_t>(indices.size());
    const auto& centroids = codebook.centroids;
    const size_t num_centroids = centroids.size();

    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) {
        size_t idx = static_cast<size_t>(in[i]);
        if (idx >= num_centroids) idx = num_centroids - 1;
        out[i] = centroids[idx];
    }

    return mlx::core::array(out.data(), {static_cast<int>(n)});
}

bool validate_codebook(const Codebook& codebook) {
    const auto& c = codebook.centroids;
    const size_t expected_count = static_cast<size_t>(1) << codebook.bits;

    if (c.size() != expected_count) return false;

    // Centroids must be strictly sorted
    for (size_t i = 1; i < c.size(); i++) {
        if (c[i] <= c[i - 1]) return false;
    }

    // Boundaries must exist and be midpoints of adjacent centroids
    const auto& b = codebook.boundaries;
    if (b.size() != c.size() - 1) return false;
    for (size_t i = 0; i < b.size(); i++) {
        float expected_mid = (c[i] + c[i + 1]) * 0.5f;
        if (std::abs(b[i] - expected_mid) > 1e-5f) return false;
    }

    // Symmetry: centroid[i] == -centroid[N-1-i]. Both the analytical codebook
    // and the data-fitted codebook build their negative half by negating the
    // positive half, so this relation holds exactly at the bit level and
    // tolerance is neither required nor desirable here — any non-zero
    // deviation indicates a construction bug.
    for (size_t i = 0; i < c.size() / 2; i++) {
        if (c[i] + c[c.size() - 1 - i] != 0.0f) return false;
    }

    return true;
}

} // namespace turboquant
