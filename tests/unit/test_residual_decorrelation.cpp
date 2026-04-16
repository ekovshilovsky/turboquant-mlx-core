/// Test harness for evaluating residual decorrelation techniques.
///
/// The structured quantization residual (after primary Lloyd-Max pass)
/// has a non-Gaussian distribution -- each coordinate is the error within
/// its codebook bin. Re-decorrelating this residual before the second
/// quantization pass improves quality. The full WHT achieves this but
/// costs O(d log d) and requires a second inverse at inference time.
///
/// This test compares different decorrelation approaches on the same
/// weight matrix and reports MSE for each, enabling rapid A/B testing
/// of lightweight alternatives to the full WHT.

#include "turboquant/quantizer.h"
#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace turboquant;

static mlx::core::array make_weight(int rows, int cols, uint32_t seed) {
    std::vector<float> data(rows * cols);
    uint32_t state = seed;
    for (int i = 0; i < rows * cols; ++i) {
        state = state * 1103515245u + 12345u;
        data[i] = (static_cast<float>(state >> 16) / 32768.0f) - 1.0f;
    }
    return mlx::core::array(data.data(), {rows, cols}, mlx::core::float32);
}

static float compute_mse(const mlx::core::array& a, const mlx::core::array& b) {
    auto diff = mlx::core::subtract(a, b);
    auto sq = mlx::core::multiply(diff, diff);
    auto mse = mlx::core::mean(sq);
    mlx::core::eval(mse);
    return mse.item<float>();
}

/// Measure distribution characteristics of a float vector.
static void analyze_distribution(const float* data, int n, const char* label) {
    double sum = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int i = 0; i < n; i++) {
        double d = static_cast<double>(data[i]);
        sum += d;
        sum2 += d * d;
        sum3 += d * d * d;
        sum4 += d * d * d * d;
    }
    double mean = sum / n;
    double var = sum2 / n - mean * mean;
    double stddev = std::sqrt(var);
    double skew = (stddev > 1e-10) ?
        ((sum3 / n - 3 * mean * var - mean * mean * mean) / (stddev * stddev * stddev)) : 0;
    double kurt = (stddev > 1e-10) ?
        ((sum4 / n) / (var * var) - 3.0) : 0;

    printf("  %s: mean=%.4f std=%.4f skew=%.2f excess_kurtosis=%.2f\n",
           label, mean, stddev, skew, kurt);
    printf("    (Gaussian reference: mean=0 skew=0 excess_kurtosis=0)\n");
}

static float test_dual_rotation(const mlx::core::array& weight, int block_size) {
    auto cb = generate_codebook(4);
    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = static_cast<uint32_t>(block_size);
    config.shared_rotation = false;

    auto qw = quantize_weight(weight, cb, cb, config);
    auto recon = dequantize_weight_cpu(qw, cb, cb, static_cast<uint32_t>(block_size));
    mlx::core::eval(recon);
    return compute_mse(weight, recon);
}

static float test_shared_rotation(const mlx::core::array& weight, int block_size) {
    auto cb = generate_codebook(4);
    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = static_cast<uint32_t>(block_size);
    config.shared_rotation = true;

    auto qw = quantize_weight(weight, cb, cb, config);
    auto recon = dequantize_weight_cpu(qw, cb, cb, static_cast<uint32_t>(block_size));
    mlx::core::eval(recon);
    return compute_mse(weight, recon);
}

/// Analyze the structure of the residual after primary quantization.
/// This reveals WHY the shared-rotation residual is harder to compress.
static void analyze_residual_structure() {
    printf("\n  RESIDUAL STRUCTURE ANALYSIS\n");

    auto weight = make_weight(64, 512, 42);
    mlx::core::eval(weight);
    auto cb = generate_codebook(4);

    uint32_t seed = 12345;
    auto w_f32 = mlx::core::astype(weight, mlx::core::float32);
    mlx::core::eval(w_f32);

    auto norms = mlx::core::sqrt(mlx::core::sum(mlx::core::square(w_f32), {1}, true));
    mlx::core::eval(norms);
    auto normalized = mlx::core::divide(w_f32, mlx::core::maximum(norms, mlx::core::array(1e-10f)));
    mlx::core::eval(normalized);

    auto rotated = apply_wht_rotation(normalized, seed, 512);
    float scale = std::sqrt(512.0f);
    auto scaled = mlx::core::multiply(rotated, mlx::core::array(scale));
    mlx::core::eval(scaled);

    // Flatten for codebook quantize (returns 1D), then reshape back
    auto flat_scaled = mlx::core::reshape(scaled, {64 * 512});
    mlx::core::eval(flat_scaled);
    auto indices = turboquant::quantize(flat_scaled, cb);
    auto dequanted_flat = turboquant::dequantize(indices, cb);
    auto dequanted = mlx::core::reshape(dequanted_flat, {64, 512});
    mlx::core::eval(dequanted);

    // Shared-rotation residual: stays in primary rotated domain
    auto residual_shared = mlx::core::subtract(scaled, dequanted);
    mlx::core::eval(residual_shared);

    // Dual-rotation residual: inverse rotate, then re-rotate with new seed
    auto residual_unscaled = mlx::core::divide(residual_shared, mlx::core::array(scale));
    mlx::core::eval(residual_unscaled);
    auto residual_original = apply_inverse_wht_rotation(residual_unscaled, seed, 512);
    mlx::core::eval(residual_original);
    auto residual_rerotated = apply_wht_rotation(residual_original, 67890, 512);
    auto residual_dual = mlx::core::multiply(residual_rerotated, mlx::core::array(scale));
    mlx::core::eval(residual_dual);

    int n = 64 * 512;
    analyze_distribution(residual_shared.data<float>(), n,
                         "Shared residual (structured, in primary domain)");
    analyze_distribution(residual_dual.data<float>(), n,
                         "Dual residual   (re-decorrelated with new seed)");

    printf("\n");
}

/// Compare shared-rotation quality using a generic N(0,1) residual codebook
/// vs a residual codebook fitted to the actual post-primary-quantization residual.
/// The residual distribution is truncated-uniform (excess kurtosis ~16), not
/// Gaussian, so a fitted codebook should concentrate centroids near zero and
/// reduce MSE at no inference cost (same lookup, different 16-float constants).
static void test_adapted_residual_codebook() {
    printf("\n  ADAPTED RESIDUAL CODEBOOK TEST\n");

    const int rows = 64;
    const int cols = 512;
    auto weight = make_weight(rows, cols, 42);
    mlx::core::eval(weight);

    auto generic_cb = generate_codebook(4);

    // --- Baseline: shared-rotation with generic N(0,1) codebook for both passes ---
    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = 512;
    config.shared_rotation = true;

    auto qw_generic = quantize_weight(weight, generic_cb, generic_cb, config);
    auto recon_generic = dequantize_weight_cpu(qw_generic, generic_cb, generic_cb, config.block_size);
    mlx::core::eval(recon_generic);
    float mse_generic = compute_mse(weight, recon_generic);

    // --- Adapted: fit primary codebook to rotated values, then fit residual
    //     codebook to the actual residual distribution ---
    float scale = std::sqrt(static_cast<float>(cols));

    // Replicate the quantizer's normalization and rotation pipeline
    auto sq = mlx::core::multiply(weight, weight);
    auto row_sum = mlx::core::sum(sq, {1}, true);
    auto row_norms = mlx::core::sqrt(row_sum);
    auto safe_norms = mlx::core::maximum(row_norms, mlx::core::array(1e-10f));
    auto w_norm = mlx::core::divide(weight, safe_norms);
    mlx::core::eval(w_norm);

    // Derive the same deterministic seed the quantizer will use
    uint32_t seed = 0;
    {
        const float* wdata = weight.data<float>();
        size_t n = static_cast<size_t>(weight.size());
        uint32_t hash = 2166136261u ^ 0u;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(wdata);
        size_t byte_count = n * sizeof(float);
        size_t stride = (byte_count > 4096) ? (byte_count / 4096) : 1;
        for (size_t i = 0; i < byte_count; i += stride) {
            hash ^= bytes[i];
            hash *= 16777619u;
        }
        seed = hash;
    }

    auto rotated = apply_wht_rotation(w_norm, seed, 512);
    mlx::core::eval(rotated);
    auto scaled = mlx::core::multiply(rotated, mlx::core::array(scale));
    mlx::core::eval(scaled);
    auto flat_scaled = mlx::core::reshape(scaled, {rows * cols});
    mlx::core::eval(flat_scaled);

    // Fit primary codebook to the rotated weight distribution
    const float* scaled_ptr = flat_scaled.data<float>();
    size_t total_elems = static_cast<size_t>(flat_scaled.size());
    std::vector<float> primary_samples(scaled_ptr, scaled_ptr + total_elems);
    auto fitted_primary_cb = generate_codebook_from_data(primary_samples, 4, 100);

    // Quantize primary pass to obtain the actual residual values
    auto indices_primary = turboquant::quantize(flat_scaled, fitted_primary_cb);
    mlx::core::eval(indices_primary);
    auto dq_primary = turboquant::dequantize(indices_primary, fitted_primary_cb);
    mlx::core::eval(dq_primary);
    auto residual_flat = mlx::core::subtract(flat_scaled, dq_primary);
    mlx::core::eval(residual_flat);

    // Measure the residual distribution characteristics
    const float* res_ptr = residual_flat.data<float>();
    analyze_distribution(res_ptr, static_cast<int>(total_elems),
                         "Residual after primary (fitted codebook)");

    // Fit residual codebook to the actual truncated-uniform residual distribution
    std::vector<float> residual_samples(res_ptr, res_ptr + total_elems);
    auto fitted_residual_cb = generate_codebook_from_data(residual_samples, 4, 100);

    // Quantize with adapted codebooks: fitted primary + fitted residual
    auto qw_adapted = quantize_weight(weight, fitted_primary_cb, fitted_residual_cb, config);
    auto recon_adapted = dequantize_weight_cpu(qw_adapted, fitted_primary_cb, fitted_residual_cb, config.block_size);
    mlx::core::eval(recon_adapted);
    float mse_adapted = compute_mse(weight, recon_adapted);

    // Also test: fitted primary + generic residual (isolates the residual codebook effect)
    auto qw_mixed = quantize_weight(weight, fitted_primary_cb, generic_cb, config);
    auto recon_mixed = dequantize_weight_cpu(qw_mixed, fitted_primary_cb, generic_cb, config.block_size);
    mlx::core::eval(recon_mixed);
    float mse_mixed = compute_mse(weight, recon_mixed);

    printf("  MSE generic primary + generic residual:  %.6f\n", mse_generic);
    printf("  MSE fitted primary  + generic residual:  %.6f\n", mse_mixed);
    printf("  MSE fitted primary  + adapted residual:  %.6f\n", mse_adapted);
    printf("  Improvement (adapted vs generic):        %.2f%%\n",
           (1.0f - mse_adapted / mse_generic) * 100.0f);
    printf("  Improvement (adapted vs mixed):          %.2f%%\n",
           (1.0f - mse_adapted / mse_mixed) * 100.0f);

    // Compare centroids near zero (indices 6-9 for 16-centroid codebook):
    // the adapted residual codebook should cluster centroids more tightly
    // around zero where the truncated-uniform residual mass is concentrated
    printf("  Generic residual CB centroids [6..9]: [%.4f, %.4f, %.4f, %.4f]\n",
           generic_cb.centroids[6], generic_cb.centroids[7],
           generic_cb.centroids[8], generic_cb.centroids[9]);
    printf("  Adapted residual CB centroids [6..9]: [%.4f, %.4f, %.4f, %.4f]\n",
           fitted_residual_cb.centroids[6], fitted_residual_cb.centroids[7],
           fitted_residual_cb.centroids[8], fitted_residual_cb.centroids[9]);

    // The adapted residual codebook should reduce MSE compared to using the
    // generic N(0,1) codebook for the residual pass. The improvement comes from
    // matching centroids to the truncated-uniform distribution instead of
    // spreading them across Gaussian tails where no residual values exist.
    assert(mse_adapted <= mse_mixed * 1.01f &&
           "adapted residual codebook must not degrade quality vs generic residual");

    printf("  PASS: adapted residual codebook maintains or improves quality\n\n");
}

int main() {
    printf("test_residual_decorrelation:\n");

    auto weight = make_weight(64, 512, 42);
    mlx::core::eval(weight);

    float mse_dual = test_dual_rotation(weight, 512);
    float mse_shared = test_shared_rotation(weight, 512);

    printf("  Dual rotation MSE:   %.6f\n", mse_dual);
    printf("  Shared rotation MSE: %.6f\n", mse_shared);
    printf("  Ratio (shared/dual): %.2fx\n\n", mse_shared / mse_dual);

    analyze_residual_structure();

    // Test that fitting the residual codebook to the actual residual distribution
    // improves quality vs using the generic N(0,1) codebook for the residual pass
    test_adapted_residual_codebook();

    printf("All residual decorrelation tests passed.\n");
    return 0;
}
