#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace turboquant;

/// Abort with a diagnostic message when a test condition fails.
/// Replaces assert() which is disabled in Release builds (-DNDEBUG).
#define TQ_CHECK(cond, msg)                                     \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "FAIL: %s (at %s:%d)\n",           \
                    (msg), __FILE__, __LINE__);                 \
            abort();                                            \
        }                                                       \
    } while (0)

/// Generate a deterministic pseudo-random weight matrix for reproducible tests.
/// Uses a simple linear congruential generator seeded from the provided value.
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

/// Validate that GPU dequantization matches the CPU reference for a given
/// weight shape and block size. Quantizes a random matrix, dequantizes via
/// both CPU and GPU paths, and verifies the maximum absolute difference is
/// within the specified tolerance.
static void validate_gpu_vs_cpu(
    int out_features,
    int in_features,
    uint32_t bs,
    float tolerance,
    const char* label) {

    auto weight = make_test_weight(out_features, in_features, 42u + bs);
    auto primary_cb = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = bs;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    auto cpu_result = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(cpu_result);

    auto gpu_result = dequantize_weight_gpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(gpu_result);

    TQ_CHECK(cpu_result.shape(0) == gpu_result.shape(0),
             "output row count mismatch between CPU and GPU");
    TQ_CHECK(cpu_result.shape(1) == gpu_result.shape(1),
             "output column count mismatch between CPU and GPU");
    TQ_CHECK(cpu_result.shape(0) == out_features,
             "unexpected output row count");
    TQ_CHECK(cpu_result.shape(1) == in_features,
             "unexpected output column count");

    float diff = max_abs_diff(cpu_result, gpu_result);
    printf("    %s: max abs diff = %.6f (tolerance = %.4f)\n", label, diff, tolerance);
    TQ_CHECK(diff < tolerance, "GPU output exceeds tolerance vs CPU reference");
}

/// GPU dequantization of a 32x128 matrix with block_size=128 must produce
/// output that matches the CPU reference within floating-point tolerance.
static void test_gpu_matches_cpu_reference() {
    validate_gpu_vs_cpu(32, 128, 128, 0.01f, "32x128 bs=128");
    printf("  PASS: GPU dequant matches CPU reference (32x128, block_size=128)\n");
}

/// Non-power-of-2 output dimension (384 = 3*128) with block_size=128
/// exercises the grid dispatch for matrices whose row count is not a
/// convenient power of two.
static void test_non_power_of_2_dims() {
    validate_gpu_vs_cpu(24, 384, 128, 0.01f, "24x384 bs=128");
    printf("  PASS: non-power-of-2 dimensions (24x384, block_size=128)\n");
}

/// Primary-only quantization (residual_bits=0) should also produce matching
/// GPU and CPU results, verifying the residual detection logic.
static void test_primary_only_quantization() {
    int out_features = 16;
    int in_features = 256;
    uint32_t bs = 128;

    auto weight = make_test_weight(out_features, in_features, 99u);
    auto primary_cb = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 0;
    config.block_size = bs;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    auto cpu_result = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(cpu_result);

    auto gpu_result = dequantize_weight_gpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(gpu_result);

    float diff = max_abs_diff(cpu_result, gpu_result);
    printf("    primary-only: max abs diff = %.6f\n", diff);
    TQ_CHECK(diff < 0.01f, "primary-only GPU output exceeds tolerance vs CPU reference");
    printf("  PASS: primary-only quantization matches CPU reference\n");
}

/// Validate that the fused dequant-matmul kernel produces output matching the
/// unfused reference (GPU dequant + matmul) within the tolerance budget that
/// accounts for float16 accumulation differences in the fused path.
static void validate_fused_vs_unfused(
    int out_features,
    int in_features,
    int batch,
    uint32_t bs,
    float tolerance,
    const char* label) {

    auto weight = make_test_weight(out_features, in_features, 42u + bs);
    auto primary_cb = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = bs;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);

    // Reference path: full dequant on CPU followed by float32 matmul
    auto weight_dequant = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);
    mlx::core::eval(weight_dequant);

    auto input = make_test_weight(batch, in_features, 99u);
    auto ref_output = mlx::core::matmul(input, mlx::core::transpose(weight_dequant));
    mlx::core::eval(ref_output);

    // Fused path: single Metal dispatch performing dequant + matmul together
    auto fused_output = fused_dequant_matmul(qw, primary_cb, residual_cb, bs, input);
    mlx::core::eval(fused_output);

    // Cast the fused float16 output to float32 for comparison
    auto fused_f32 = mlx::core::astype(fused_output, mlx::core::float32);
    mlx::core::eval(fused_f32);

    TQ_CHECK(fused_f32.shape(0) == batch,
             "fused output batch dimension mismatch");
    TQ_CHECK(fused_f32.shape(1) == out_features,
             "fused output feature dimension mismatch");

    float diff = max_abs_diff(fused_f32, ref_output);
    printf("    %s: max abs diff = %.6f (tolerance = %.4f)\n", label, diff, tolerance);
    TQ_CHECK(diff < tolerance, "fused dequant-matmul output exceeds tolerance vs unfused reference");
}

/// Fused dequant-matmul must produce output matching the CPU dequant + matmul
/// reference for a standard matrix size with block_size=128.
static void test_fused_dequant_matmul_correctness() {
    validate_fused_vs_unfused(32, 128, 1, 128, 0.15f, "fused 32x128 batch=1");
    validate_fused_vs_unfused(32, 128, 4, 128, 0.15f, "fused 32x128 batch=4");
    printf("  PASS: fused dequant-matmul matches unfused reference\n");
}

/// Fused path with a larger block_size exercises the WHT butterfly at a
/// different depth and verifies the sign buffer is correctly sized.
static void test_fused_dequant_matmul_large_block() {
    validate_fused_vs_unfused(16, 512, 1, 256, 0.15f, "fused 16x512 bs=256");
    printf("  PASS: fused dequant-matmul with block_size=256\n");
}

int main() {
    printf("test_dequant_kernel (integration):\n");
    test_gpu_matches_cpu_reference();
    test_non_power_of_2_dims();
    test_primary_only_quantization();
    test_fused_dequant_matmul_correctness();
    test_fused_dequant_matmul_large_block();
    printf("All dequant kernel integration tests passed.\n");
    return 0;
}
