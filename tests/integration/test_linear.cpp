#include "turboquant/linear.h"
#include "turboquant/codebook.h"
#include "turboquant/dequantizer.h"
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

/// Generate a deterministic pseudo-random matrix for reproducible tests.
/// Uses a simple linear congruential generator seeded from the provided value.
static mlx::core::array make_test_matrix(int rows, int cols, uint32_t seed) {
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

/// Verify that TurboQuantLinear::forward produces the correct output shape
/// for the given batch size, in_features, and out_features dimensions.
static void validate_output_shape(int batch, int in_features, int out_features,
                                  const char* label) {
    auto weight = make_test_matrix(out_features, in_features, 7u);
    auto primary_cb = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = 128;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);
    TurboQuantLinear linear(in_features, out_features, qw, primary_cb, residual_cb,
                            config.block_size);

    auto input = make_test_matrix(batch, in_features, 13u);
    auto output = linear.forward(input);
    mlx::core::eval(output);

    TQ_CHECK(output.ndim() == 2, "forward output must be 2-dimensional");
    TQ_CHECK(output.shape(0) == batch,
             "forward output batch dimension must match input");
    TQ_CHECK(output.shape(1) == out_features,
             "forward output feature dimension must match out_features");

    printf("    %s: output shape [%d, %d] correct\n",
           label, static_cast<int>(output.shape(0)),
           static_cast<int>(output.shape(1)));
}

/// Verify that TurboQuantLinear::forward produces output consistent with a
/// manual dequant + matmul reference for a given batch size.
/// The maximum absolute error should be below the specified tolerance to
/// account for floating-point accumulation differences.
static void validate_forward_correctness(int batch, int in_features, int out_features,
                                         float tolerance, const char* label) {
    auto weight = make_test_matrix(out_features, in_features, 42u);
    auto primary_cb = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig config{};
    config.primary_bits = 4;
    config.residual_bits = 4;
    config.block_size = 128;
    config.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, config);
    TurboQuantLinear linear(in_features, out_features, qw, primary_cb, residual_cb,
                            config.block_size);

    // Build the reference output: CPU dequant followed by manual matmul.
    // weight_dequant is [out_features, in_features]; input is [batch, in_features].
    // output_ref = input @ weight_dequant^T  =>  [batch, out_features]
    auto input = make_test_matrix(batch, in_features, 99u);
    auto weight_dequant = dequantize_weight_cpu(qw, primary_cb, residual_cb,
                                                config.block_size);
    mlx::core::eval(weight_dequant);

    auto output_ref = mlx::core::matmul(input, mlx::core::transpose(weight_dequant));
    mlx::core::eval(output_ref);

    // TurboQuantLinear forward uses the fused dequant-matmul path which
    // returns float16; cast to float32 for comparison with the reference.
    auto output_tq = linear.forward(input);
    mlx::core::eval(output_tq);
    auto output_tq_f32 = mlx::core::astype(output_tq, mlx::core::float32);
    mlx::core::eval(output_tq_f32);

    float diff = max_abs_diff(output_tq_f32, output_ref);
    printf("    %s (batch=%d): max abs diff = %.6f (tolerance = %.4f)\n",
           label, batch, diff, tolerance);
    TQ_CHECK(diff < tolerance,
             "TurboQuantLinear forward output exceeds tolerance vs CPU reference");
}

/// Output shape must be [batch, out_features] for any valid input shape.
static void test_output_shape() {
    validate_output_shape(1,  128, 64, "batch=1");
    validate_output_shape(8,  128, 64, "batch=8");
    printf("  PASS: output shape is correct for batch=1 and batch=8\n");
}

/// TurboQuantLinear forward must match the CPU dequant + matmul reference
/// within the expected quantization + float16 accumulation error budget.
/// The fused kernel accumulates in float32 but outputs float16, so a wider
/// tolerance accounts for the reduced output precision.
static void test_forward_correctness() {
    validate_forward_correctness(1, 128, 64, 0.15f, "correctness");
    printf("  PASS: forward output matches CPU reference (batch=1, tol=0.15)\n");
}

/// Both batch=1 (single-sample) and batch=8 (mini-batch) must produce
/// numerically correct results to confirm batch dimension handling is correct.
static void test_batch_dimension() {
    validate_forward_correctness(1, 128, 64, 0.15f, "batch=1");
    validate_forward_correctness(8, 128, 64, 0.15f, "batch=8");
    printf("  PASS: batch=1 and batch=8 both produce correct results\n");
}

int main() {
    printf("test_linear (integration):\n");
    test_output_shape();
    test_forward_correctness();
    test_batch_dimension();
    printf("All linear integration tests passed.\n");
    return 0;
}
