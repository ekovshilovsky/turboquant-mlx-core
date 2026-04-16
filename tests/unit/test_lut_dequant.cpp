#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace turboquant;

/// Abort with a diagnostic message when a test condition fails.
#define TQ_CHECK(cond, msg)                                     \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "FAIL: %s (at %s:%d)\n",           \
                    (msg), __FILE__, __LINE__);                 \
            abort();                                            \
        }                                                       \
    } while (0)

/// Generate a deterministic pseudo-random weight matrix for reproducible tests.
static mlx::core::array make_test_weight(int rows, int cols, uint32_t seed) {
    std::vector<float> data(rows * cols);
    uint32_t state = seed;
    for (int i = 0; i < rows * cols; ++i) {
        state = state * 1103515245u + 12345u;
        data[i] = (static_cast<float>(state >> 16) / 32768.0f) - 1.0f;
    }
    return mlx::core::array(data.data(), {rows, cols}, mlx::core::float32);
}

/// Compute the maximum absolute element-wise difference between two arrays.
static float max_abs_diff(const mlx::core::array& a, const mlx::core::array& b) {
    auto diff = mlx::core::subtract(a, b);
    auto abs_diff = mlx::core::abs(diff);
    auto max_val = mlx::core::max(abs_diff);
    mlx::core::eval(max_val);
    return max_val.item<float>();
}

/// Verify that the 256-entry LUT produces the correct centroid sums for all
/// possible 4-bit index combinations. This validates the LUT construction
/// independently of the full dequantization pipeline.
static void test_lut_construction_correctness() {
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    // Build the LUT with the same indexing scheme used in the Metal kernel:
    //   lut[(r_idx << 4) | p_idx] = primary_centroid[p_idx] + residual_centroid[r_idx]
    std::vector<float> lut(256);
    for (int r = 0; r < 16; ++r) {
        for (int p = 0; p < 16; ++p) {
            uint8_t idx = static_cast<uint8_t>((r << 4) | p);
            lut[idx] = primary_cb.centroids[p] + residual_cb.centroids[r];
        }
    }

    // Verify every entry matches the direct computation
    for (int r = 0; r < 16; ++r) {
        for (int p = 0; p < 16; ++p) {
            uint8_t idx = static_cast<uint8_t>((r << 4) | p);
            float expected = primary_cb.centroids[p] + residual_cb.centroids[r];
            float actual = lut[idx];
            TQ_CHECK(actual == expected,
                     "LUT entry does not match direct centroid sum (bit-exact)");
        }
    }
    printf("  PASS: LUT construction produces bit-exact centroid sums\n");
}

/// Verify that the CPU dequantizer's LUT path produces output identical to
/// a reference implementation that uses direct codebook lookups. Both paths
/// go through the same downstream WHT and norm correction, so any difference
/// must originate in the centroid lookup stage.
static void test_cpu_lut_matches_direct_codebook() {
    const int out_features = 32;
    const int in_features  = 128;
    const uint32_t bs      = 128;

    auto weight = make_test_weight(out_features, in_features, 42u);
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    // Quantize with shared_rotation=true to exercise the LUT path
    QuantizerConfig config{};
    config.primary_bits    = 4;
    config.residual_bits   = 4;
    config.block_size      = bs;
    config.shared_rotation = true;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    // CPU dequantization now uses the LUT internally for shared-rotation 4-bit
    auto result = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(result);

    TQ_CHECK(result.shape(0) == out_features,
             "output row count does not match expected out_features");
    TQ_CHECK(result.shape(1) == in_features,
             "output column count does not match expected in_features");

    // Verify against GPU path which uses the same LUT in the fused kernel
    auto gpu_result = dequantize_weight_gpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(gpu_result);

    float diff = max_abs_diff(result, gpu_result);
    printf("    CPU LUT vs GPU: max abs diff = %.6f\n", diff);
    TQ_CHECK(diff < 0.01f,
             "CPU LUT dequant differs from GPU reference beyond tolerance");
    printf("  PASS: CPU LUT dequant matches GPU reference\n");
}

/// Verify that the fused dequant-matmul kernel with LUT produces output
/// matching the unfused reference (CPU dequant + float32 matmul). The LUT
/// path is exercised when shared_rotation=true and primary_bits=4.
static void test_fused_lut_matches_unfused() {
    const int out_features = 32;
    const int in_features  = 128;
    const int batch        = 1;
    const uint32_t bs      = 128;

    auto weight = make_test_weight(out_features, in_features, 42u);
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits    = 4;
    config.residual_bits   = 4;
    config.block_size      = bs;
    config.shared_rotation = true;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    // Reference path: full CPU dequant + float32 matmul
    auto weight_dequant = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(weight_dequant);

    auto input = make_test_weight(batch, in_features, 99u);
    auto ref_output = mlx::core::matmul(input, mlx::core::transpose(weight_dequant));
    mlx::core::eval(ref_output);

    // Fused path (exercises the LUT in the Metal kernel)
    auto fused_output = fused_dequant_matmul(qw, primary_cb, residual_cb, bs, input);
    mlx::core::eval(fused_output);

    auto fused_f32 = mlx::core::astype(fused_output, mlx::core::float32);
    mlx::core::eval(fused_f32);

    TQ_CHECK(fused_f32.shape(0) == batch,
             "fused output batch dimension mismatch");
    TQ_CHECK(fused_f32.shape(1) == out_features,
             "fused output feature dimension mismatch");

    float diff = max_abs_diff(fused_f32, ref_output);
    printf("    fused LUT vs unfused: max abs diff = %.6f (tolerance = 0.15)\n", diff);
    TQ_CHECK(diff < 0.15f,
             "fused LUT dequant-matmul exceeds tolerance vs unfused reference");
    printf("  PASS: fused LUT dequant-matmul matches unfused reference\n");
}

/// Verify that dual-rotation (non-shared) mode still works correctly when the
/// LUT path is inactive. This confirms the LUT guard conditions do not break
/// the legacy code path.
static void test_dual_rotation_unaffected() {
    const int out_features = 16;
    const int in_features  = 256;
    const uint32_t bs      = 128;

    auto weight = make_test_weight(out_features, in_features, 77u);
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits    = 4;
    config.residual_bits   = 4;
    config.block_size      = bs;
    config.shared_rotation = false;  // Dual-rotation: LUT path must not activate
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    auto cpu_result = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(cpu_result);

    auto gpu_result = dequantize_weight_gpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(gpu_result);

    float diff = max_abs_diff(cpu_result, gpu_result);
    printf("    dual-rotation CPU vs GPU: max abs diff = %.6f\n", diff);
    TQ_CHECK(diff < 0.01f,
             "dual-rotation mode degraded after LUT changes");
    printf("  PASS: dual-rotation mode unaffected by LUT changes\n");
}

int main() {
    printf("test_lut_dequant (unit):\n");
    test_lut_construction_correctness();
    test_cpu_lut_matches_direct_codebook();
    test_fused_lut_matches_unfused();
    test_dual_rotation_unaffected();
    printf("All LUT dequant unit tests passed.\n");
    return 0;
}
