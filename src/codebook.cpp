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

} // namespace

Codebook generate_codebook(uint8_t bits) {
    Codebook cb;
    switch (bits) {
        case 1: cb = build_from_centroids(kCentroids1Bit, 2,  bits); break;
        case 2: cb = build_from_centroids(kCentroids2Bit, 4,  bits); break;
        case 3: cb = build_from_centroids(kCentroids3Bit, 8,  bits); break;
        case 4: cb = build_from_centroids(kCentroids4Bit, 16, bits); break;
        case 5: cb = build_from_centroids(kCentroids5Bit, 32, bits); break;
        default:
            throw std::invalid_argument("Unsupported bit width; must be 1-5");
    }
    cb.origin = CodebookOrigin::Analytical;
    return cb;
}

/// CPU-only Lloyd-Max iteration for small datasets. Uses binary search over
/// boundaries for bin assignment, which is efficient for datasets below the
/// GPU dispatch threshold where kernel launch overhead would dominate.
///
/// Fits all N centroids directly against the raw data so the resulting
/// codebook can capture any skew in the empirical distribution. Downstream
/// code does not assume symmetry of empirical codebooks.
///
/// External linkage is retained (no `static`) so the CPU and GPU paths can
/// be invoked directly from tests that verify both fits produce equivalent
/// codebooks on identical input. The public entry point is still
/// `generate_codebook_from_data`, which dispatches by sample count.
Codebook generate_codebook_from_data_cpu(
    const std::vector<float>& data, uint8_t bits, int iterations) {

    Codebook cb = generate_codebook(bits);
    const size_t num_centroids = cb.centroids.size();
    const size_t n = data.size();

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i + 1 < num_centroids; ++i) {
            cb.boundaries[i] = (cb.centroids[i] + cb.centroids[i + 1]) * 0.5f;
        }

        std::vector<double> bin_sum(num_centroids, 0.0);
        std::vector<size_t> bin_count(num_centroids, 0);

        for (size_t i = 0; i < n; ++i) {
            auto it = std::lower_bound(cb.boundaries.begin(), cb.boundaries.end(), data[i]);
            size_t idx = static_cast<size_t>(it - cb.boundaries.begin());
            if (idx >= num_centroids) idx = num_centroids - 1;
            bin_sum[idx] += static_cast<double>(data[i]);
            bin_count[idx]++;
        }

        float max_shift = 0.0f;
        for (size_t i = 0; i < num_centroids; ++i) {
            if (bin_count[i] > 0) {
                float new_centroid = static_cast<float>(bin_sum[i] / static_cast<double>(bin_count[i]));
                float shift = std::abs(new_centroid - cb.centroids[i]);
                if (shift > max_shift) max_shift = shift;
                cb.centroids[i] = new_centroid;
            }
        }

        if (max_shift < kLloydMaxShiftThreshold) break;
    }

    // Recompute final boundaries from the converged centroids so they remain
    // consistent with the stored centroid values.
    for (size_t i = 0; i + 1 < num_centroids; ++i) {
        cb.boundaries[i] = (cb.centroids[i] + cb.centroids[i + 1]) * 0.5f;
    }

    cb.origin = CodebookOrigin::Empirical;
    return cb;
}

/// GPU-accelerated Lloyd-Max iteration using MLX array operations.
/// For large datasets (millions of weight values), the per-bin masked
/// sum and count operations run on GPU through MLX's lazy evaluation,
/// avoiding the serial binary search loop over every element.
///
/// Fits all N centroids directly against the raw data so the resulting
/// codebook can capture any skew in the empirical distribution. Downstream
/// code does not assume symmetry of empirical codebooks.
///
/// External linkage is retained (no `static`) so the CPU and GPU paths can
/// be invoked directly from tests that verify both fits produce equivalent
/// codebooks on identical input. The public entry point is still
/// `generate_codebook_from_data`, which dispatches by sample count.
Codebook generate_codebook_from_data_gpu(
    const std::vector<float>& data, uint8_t bits, int iterations) {

    Codebook cb = generate_codebook(bits);
    const int num_centroids = static_cast<int>(cb.centroids.size());

    // Transfer the data to GPU once; reused across all iterations.
    auto data_arr = mlx::core::array(
        data.data(),
        {static_cast<int>(data.size())},
        mlx::core::float32);

    for (int iter = 0; iter < iterations; ++iter) {
        // Recompute boundaries as midpoints between adjacent centroids.
        for (int i = 0; i + 1 < num_centroids; ++i) {
            cb.boundaries[i] = (cb.centroids[i] + cb.centroids[i + 1]) * 0.5f;
        }

        // Build boundary array on GPU for vectorized bin assignment.
        // For typical bit widths (2-5 bits = 4-32 centroids), the boundary
        // array is tiny and the overhead is negligible compared to the
        // reduction over millions of data elements.
        auto boundaries_arr = mlx::core::array(
            cb.boundaries.data(),
            {static_cast<int>(cb.boundaries.size())},
            mlx::core::float32);

        // Compute per-bin masked sums and counts using GPU reductions.
        // Each bin is defined by its lower and upper boundaries. The
        // comparison chain produces a boolean mask per bin, which is
        // multiplied element-wise with the data to produce masked values
        // for summation. This replaces the CPU's per-element binary search
        // with N parallel GPU comparisons (N = number of bins).
        //
        // MLX reductions run in float32 with implementation-defined summation
        // order (typically a tree reduction). The CPU path uses double-precision
        // accumulation for stability. Expect ~1-ULP drift between the two paths
        // at the centroid level — still well under float16 storage precision.
        std::vector<mlx::core::array> bin_sums;
        std::vector<mlx::core::array> bin_counts;
        bin_sums.reserve(static_cast<size_t>(num_centroids));
        bin_counts.reserve(static_cast<size_t>(num_centroids));

        for (int bin = 0; bin < num_centroids; ++bin) {
            mlx::core::array mask(0.0f);
            if (bin == 0 && bin == num_centroids - 1) {
                // Single-bin edge case: all elements belong to this bin.
                mask = mlx::core::astype(
                    mlx::core::ones_like(data_arr), mlx::core::bool_);
            } else if (bin == 0) {
                // First bin: all elements below the first boundary.
                auto hi = mlx::core::take(boundaries_arr, 0);
                mask = mlx::core::less(data_arr, hi);
            } else if (bin == num_centroids - 1) {
                // Last bin: all elements at or above the last boundary.
                auto lo = mlx::core::take(boundaries_arr, bin - 1);
                mask = mlx::core::greater_equal(data_arr, lo);
            } else {
                // Interior bin: elements in [boundaries[bin-1], boundaries[bin]).
                auto lo = mlx::core::take(boundaries_arr, bin - 1);
                auto hi = mlx::core::take(boundaries_arr, bin);
                auto ge = mlx::core::greater_equal(data_arr, lo);
                auto lt = mlx::core::less(data_arr, hi);
                mask = mlx::core::logical_and(ge, lt);
            }

            auto mask_f = mlx::core::astype(mask, mlx::core::float32);
            auto masked_data = mlx::core::multiply(data_arr, mask_f);
            bin_sums.push_back(mlx::core::sum(masked_data));
            bin_counts.push_back(mlx::core::sum(mask_f));
        }

        // Materialize all bin reductions in a single GPU dispatch to maximize
        // occupancy and minimize synchronization points.
        std::vector<mlx::core::array> to_materialize;
        to_materialize.reserve(static_cast<size_t>(num_centroids) * 2);
        for (int bin = 0; bin < num_centroids; ++bin) {
            to_materialize.push_back(bin_sums[bin]);
            to_materialize.push_back(bin_counts[bin]);
        }
        mlx::core::eval(to_materialize);

        // Update centroids from the GPU-computed sums and counts.
        // Empty bins retain their previous centroid to maintain coverage.
        float max_shift = 0.0f;
        for (int bin = 0; bin < num_centroids; ++bin) {
            float count = bin_counts[bin].item<float>();
            if (count > 0.0f) {
                float new_centroid = bin_sums[bin].item<float>() / count;
                float shift = std::abs(new_centroid - cb.centroids[bin]);
                if (shift > max_shift) max_shift = shift;
                cb.centroids[bin] = new_centroid;
            }
        }

        // Terminate early when all centroids have stabilized. WHT-rotated
        // weight distributions are near-Gaussian, so convergence from the
        // N(0,1) initial guess typically completes in 20-30 iterations.
        if (max_shift < kLloydMaxShiftThreshold) break;
    }

    // Recompute final boundaries from the converged centroids so they remain
    // consistent with the stored centroid values.
    for (int i = 0; i + 1 < num_centroids; ++i) {
        cb.boundaries[i] = (cb.centroids[i] + cb.centroids[i + 1]) * 0.5f;
    }

    cb.origin = CodebookOrigin::Empirical;
    return cb;
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

    if (codebook.origin == CodebookOrigin::Analytical) {
        // Strict symmetry: analytical codebooks model N(0,1) and are
        // symmetric by construction. Exact bit-level equality enforced.
        for (size_t i = 0; i < c.size() / 2; i++) {
            if (c[i] + c[c.size() - 1 - i] != 0.0f) return false;
        }
    }
    // Empirical codebooks may legitimately capture distributional skew;
    // no symmetry constraint. Sorted centroids + valid boundaries are
    // checked above and remain required.

    return true;
}

} // namespace turboquant
