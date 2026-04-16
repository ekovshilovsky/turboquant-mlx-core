#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

void bench_dequant() {
    printf("bench_dequant: Metal kernel throughput\n");
    printf("  Matrix: 4096 x 4096, 4+4 bit, block_size=512\n");

    // Build codebooks for the 4-bit primary and 4-bit residual passes
    auto primary_cb  = turboquant::generate_codebook(4);
    auto residual_cb = turboquant::generate_codebook(4);

    // Create a reproducible random 4096x4096 weight matrix (float32)
    const int rows = 4096;
    const int cols = 4096;
    std::mt19937 rng(0xDEADBEEF);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(cols)));

    std::vector<float> weight_data(rows * cols);
    for (auto& v : weight_data) {
        v = dist(rng);
    }
    auto weight = mlx::core::array(weight_data.data(), {rows, cols}, mlx::core::float32);
    // Force host-to-device transfer before timing begins
    mlx::core::eval(weight);

    // Quantize once offline — quantization time is excluded from the benchmark.
    // shared_rotation=true (the default) enables the single-WHT dequant path.
    turboquant::QuantizerConfig cfg;
    cfg.primary_bits    = 4;
    cfg.residual_bits   = 4;
    cfg.block_size      = 512;
    cfg.shared_rotation = true;
    cfg.norm_correction = true;

    auto qw = turboquant::quantize_weight(weight, primary_cb, residual_cb, cfg);

    // Warm-up: one iteration to populate GPU caches and JIT-compile Metal kernels
    {
        auto result = turboquant::dequantize_weight_gpu(qw, primary_cb, residual_cb, 512);
        // Synchronize with the GPU before starting the timed loop
        mlx::core::eval(result);
    }

    // Timed benchmark: 10 synchronous GPU dequantization passes
    const int iterations = 10;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = turboquant::dequantize_weight_gpu(qw, primary_cb, residual_cb, 512);
        // Synchronize with the GPU to obtain a wall-clock-accurate measurement
        mlx::core::eval(result);
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double ms_per_iter = total_ms / iterations;

    // Effective throughput: bytes written to the output float32 matrix
    double output_bytes = static_cast<double>(rows) * cols * sizeof(float);
    double gb_per_s = (output_bytes / (ms_per_iter * 1e-3)) / 1e9;

    printf("  Iterations      : %d\n", iterations);
    printf("  Total time      : %.2f ms\n", total_ms);
    printf("  Time / iter     : %.3f ms\n", ms_per_iter);
    printf("  Throughput      : %.2f GB/s\n", gb_per_s);
    printf("\n");
}
