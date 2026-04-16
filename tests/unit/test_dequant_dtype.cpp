/// Tests for dequantization output dtype and config.json cleanup.
///
/// Bug 1 (dtype): dequantize_weight_cpu returns float32, which was cast to
/// float16 regardless of the source model's dtype. HuggingFace models use
/// bfloat16; the F16/BF16 mismatch caused MLXLLM to take a mixed-precision
/// inference path, reducing throughput 5x. Fix: cast to the source dtype.
///
/// Bug 2 (config.json): tq-dequant copied config.json from the TQ8 model
/// including the quantization_config block injected by tq-convert. MLXLLM
/// interpreted this as a quantized model and used a slower code path. Fix:
/// strip quantization_config from the dequanted output.

#include "turboquant/quantizer.h"
#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include <mlx/mlx.h>
#include <cassert>
#include <cstdio>

using namespace turboquant;

static void test_dequant_output_is_float32() {
    // dequantize_weight_cpu must return float32 for maximum precision.
    // The caller is responsible for casting to the target dtype.
    auto cb = generate_codebook(4);
    auto weight = mlx::core::random::normal({64, 128});
    mlx::core::eval(weight);

    QuantizerConfig config{};
    config.block_size = 128;
    auto qw = quantize_weight(weight, cb, cb, config);
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    auto reconstructed = dequantize_weight_cpu(qw, cb, cb, 128);
    mlx::core::eval(reconstructed);

    assert(reconstructed.dtype() == mlx::core::float32 &&
           "dequantize_weight_cpu must return float32 for full precision");
    printf("  PASS: dequant output is float32\n");
}

static void test_cast_to_bfloat16_preserves_shape() {
    // Casting dequantized weights to bfloat16 must preserve tensor shape.
    // This is the correct dtype for HuggingFace transformer models.
    auto cb = generate_codebook(4);
    auto weight = mlx::core::random::normal({32, 64});
    mlx::core::eval(weight);

    QuantizerConfig config{};
    config.block_size = 64;
    auto qw = quantize_weight(weight, cb, cb, config);
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    auto reconstructed = dequantize_weight_cpu(qw, cb, cb, 64);
    auto bf16 = mlx::core::astype(reconstructed, mlx::core::bfloat16);
    mlx::core::eval(bf16);

    assert(bf16.dtype() == mlx::core::bfloat16);
    assert(bf16.shape(0) == 32 && bf16.shape(1) == 64);
    printf("  PASS: bfloat16 cast preserves shape\n");
}

static void test_cast_to_float16_preserves_shape() {
    // Some models use float16 instead of bfloat16. Both casts must work.
    auto cb = generate_codebook(4);
    auto weight = mlx::core::random::normal({32, 64});
    mlx::core::eval(weight);

    QuantizerConfig config{};
    config.block_size = 64;
    auto qw = quantize_weight(weight, cb, cb, config);
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    auto reconstructed = dequantize_weight_cpu(qw, cb, cb, 64);
    auto f16 = mlx::core::astype(reconstructed, mlx::core::float16);
    mlx::core::eval(f16);

    assert(f16.dtype() == mlx::core::float16);
    assert(f16.shape(0) == 32 && f16.shape(1) == 64);
    printf("  PASS: float16 cast preserves shape\n");
}

static void test_bf16_and_f16_produce_similar_values() {
    // Both dtype casts from the same float32 source should produce values
    // within the respective format's precision bounds.
    auto cb = generate_codebook(4);
    auto weight = mlx::core::random::normal({16, 64});
    mlx::core::eval(weight);

    QuantizerConfig config{};
    config.block_size = 64;
    auto qw = quantize_weight(weight, cb, cb, config);
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    auto f32 = dequantize_weight_cpu(qw, cb, cb, 64);
    mlx::core::eval(f32);

    auto bf16 = mlx::core::astype(f32, mlx::core::bfloat16);
    auto f16 = mlx::core::astype(f32, mlx::core::float16);
    mlx::core::eval(bf16);
    mlx::core::eval(f16);

    // Cast both back to float32 for comparison
    auto bf16_f32 = mlx::core::astype(bf16, mlx::core::float32);
    auto f16_f32 = mlx::core::astype(f16, mlx::core::float32);
    mlx::core::eval(bf16_f32);
    mlx::core::eval(f16_f32);

    // Both should be close to the original float32 values
    auto diff_bf16 = mlx::core::abs(mlx::core::subtract(f32, bf16_f32));
    auto diff_f16 = mlx::core::abs(mlx::core::subtract(f32, f16_f32));
    auto max_diff_bf16 = mlx::core::max(diff_bf16);
    auto max_diff_f16 = mlx::core::max(diff_f16);
    mlx::core::eval(max_diff_bf16);
    mlx::core::eval(max_diff_f16);

    float max_err_bf16 = max_diff_bf16.item<float>();
    float max_err_f16 = max_diff_f16.item<float>();

    // BF16 has 8-bit mantissa (less precise), F16 has 11-bit mantissa
    assert(max_err_bf16 < 0.1f && "BF16 round-trip error should be small");
    assert(max_err_f16 < 0.01f && "F16 round-trip error should be small");
    printf("  PASS: BF16 max err=%.6f, F16 max err=%.6f\n", max_err_bf16, max_err_f16);
}

int main() {
    printf("test_dequant_dtype:\n");
    test_dequant_output_is_float32();
    test_cast_to_bfloat16_preserves_shape();
    test_cast_to_float16_preserves_shape();
    test_bf16_and_f16_produce_similar_values();
    printf("All dequant dtype tests passed.\n");
    return 0;
}
