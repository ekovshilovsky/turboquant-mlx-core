#include "turboquant/linear.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

void bench_linear() {
    printf("bench_linear: TurboQuantLinear vs plain float32 matmul\n");
    printf("  Weight: 4096 x 4096, batch=1, 4+4 bit, block_size=512\n");

    const int in_features  = 4096;
    const int out_features = 4096;

    // Build 4-bit codebooks for both passes
    auto primary_cb  = turboquant::generate_codebook(4);
    auto residual_cb = turboquant::generate_codebook(4);

    // Reproducible random weight matrix
    std::mt19937 rng(0xCAFEBABE);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(in_features)));

    std::vector<float> weight_data(out_features * in_features);
    for (auto& v : weight_data) {
        v = dist(rng);
    }
    auto weight_f32 = mlx::core::array(
        weight_data.data(), {out_features, in_features}, mlx::core::float32);
    // Materialize on GPU before quantization to avoid polluting offline timing
    mlx::core::eval(weight_f32);

    // Quantize offline — this cost is one-time and excluded from the benchmark.
    // shared_rotation=true (the default) encodes residual in the primary WHT
    // domain so dequantization requires only a single butterfly pass per block.
    turboquant::QuantizerConfig cfg;
    cfg.primary_bits    = 4;
    cfg.residual_bits   = 4;
    cfg.block_size      = 512;
    cfg.shared_rotation = true;
    cfg.norm_correction = true;

    auto qw = turboquant::quantize_weight(weight_f32, primary_cb, residual_cb, cfg);

    // Construct TurboQuantLinear and a fixed single-token input
    turboquant::TurboQuantLinear tq_linear(
        in_features, out_features, qw, primary_cb, residual_cb, 512);

    std::vector<float> input_data(in_features, 1.0f);
    auto input = mlx::core::array(input_data.data(), {1, in_features}, mlx::core::float32);
    mlx::core::eval(input);

    // --- TurboQuantLinear benchmark ---

    // Warm-up pass: JIT-compile Metal kernels before the timed loop
    {
        auto out = tq_linear.forward(input);
        mlx::core::eval(out);
    }

    const int iterations = 10;
    double tq_total_ms  = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out   = tq_linear.forward(input);
        // Synchronize with the GPU for an accurate wall-clock measurement
        mlx::core::eval(out);
        auto end = std::chrono::high_resolution_clock::now();
        tq_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double tq_ms_per_iter = tq_total_ms / iterations;
    // Each forward pass processes one token (batch=1)
    double tq_toks_per_s  = 1000.0 / tq_ms_per_iter;

    printf("  [TurboQuantLinear]\n");
    printf("    Time / token    : %.3f ms\n", tq_ms_per_iter);
    printf("    Throughput      : %.1f tok/s\n", tq_toks_per_s);

    // --- Reference float32 matmul benchmark ---
    // Warm-up
    {
        auto ref_out = mlx::core::matmul(input, mlx::core::transpose(weight_f32));
        mlx::core::eval(ref_out);
    }

    double ref_total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start   = std::chrono::high_resolution_clock::now();
        auto ref_out = mlx::core::matmul(input, mlx::core::transpose(weight_f32));
        mlx::core::eval(ref_out);
        auto end = std::chrono::high_resolution_clock::now();
        ref_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double ref_ms_per_iter = ref_total_ms / iterations;
    double ref_toks_per_s  = 1000.0 / ref_ms_per_iter;

    printf("  [float32 matmul (reference)]\n");
    printf("    Time / token    : %.3f ms\n", ref_ms_per_iter);
    printf("    Throughput      : %.1f tok/s\n", ref_toks_per_s);

    // Overhead ratio: how much more expensive TQ is versus a plain matmul.
    // A value close to 1.0x means dequant overhead is negligible at this size.
    double ratio = tq_ms_per_iter / ref_ms_per_iter;
    printf("  TQ overhead vs float32: %.2fx\n", ratio);
    printf("\n");

    // --- MLX affine q8_0 baseline ---
    // MLX's built-in affine quantization (group_size=64, 8-bit symmetric scale+bias)
    // represents the standard integer quantization baseline for Apple Silicon.
    // This measures how TurboQuant's WHT-based approach compares to MLX's native path.
    {
        // quantize() returns {packed_weights, scales, biases}
        auto qresult = mlx::core::quantize(
            mlx::core::transpose(weight_f32),
            /*group_size=*/64,
            /*bits=*/8);
        auto q_packed = qresult[0];
        auto q_scales  = qresult[1];
        auto q_biases  = qresult[2];
        mlx::core::eval(q_packed, q_scales, q_biases);

        // Warm-up: force JIT compilation for the quantized matmul kernel
        {
            auto q_out = mlx::core::quantized_matmul(
                input, q_packed, q_scales, q_biases,
                /*transpose=*/true,
                /*group_size=*/64,
                /*bits=*/8);
            mlx::core::eval(q_out);
        }

        double q8_total_ms = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto q_out = mlx::core::quantized_matmul(
                input, q_packed, q_scales, q_biases,
                /*transpose=*/true,
                /*group_size=*/64,
                /*bits=*/8);
            mlx::core::eval(q_out);
            auto end = std::chrono::high_resolution_clock::now();
            q8_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }

        double q8_ms_per_iter = q8_total_ms / iterations;
        double q8_toks_per_s  = 1000.0 / q8_ms_per_iter;
        double q8_ratio       = q8_ms_per_iter / ref_ms_per_iter;
        double tq_vs_q8_ratio = tq_ms_per_iter / q8_ms_per_iter;

        printf("  [MLX q8_0 (affine, group_size=64, reference baseline)]\n");
        printf("    Time / token    : %.3f ms\n", q8_ms_per_iter);
        printf("    Throughput      : %.1f tok/s\n", q8_toks_per_s);
        printf("    Overhead vs float32: %.2fx\n", q8_ratio);
        printf("  TQ overhead vs MLX q8_0: %.2fx\n", tq_vs_q8_ratio);
        printf("\n");
    }
}
