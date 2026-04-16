/// Verify GPU is active during inference by comparing fused Metal kernel
/// throughput against CPU-only dequantization. The fused GPU path should be
/// significantly faster than CPU dequant alone (which does not include matmul),
/// proving the Metal kernel is dispatching to GPU hardware.

#include "turboquant/turboquant.h"
#include "turboquant/dequantizer.h"
#include <mlx/mlx.h>
#include <cassert>
#include <cstdio>
#include <chrono>

using namespace turboquant;

static void test_gpu_available() {
    assert(mlx::core::metal::is_available() && "Metal GPU not available");
    assert(mlx::core::default_device() == mlx::core::Device::gpu &&
           "Default device is not GPU");
    printf("  PASS: GPU available, default device is GPU\n");
}

static void test_fused_kernel_faster_than_cpu() {
    auto cb = generate_codebook(4);
    auto weight = mlx::core::random::normal({2048, 2048});
    mlx::core::eval(weight);

    QuantizerConfig config{};
    config.block_size = 512;
    auto qw = quantize_weight(weight, cb, cb, config);
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    auto input = mlx::core::random::normal({1, 2048});
    mlx::core::eval(input);

    // Warm up both paths
    auto w_cpu = dequantize_weight_cpu(qw, cb, cb, 512);
    mlx::core::eval(w_cpu);
    auto w_gpu = fused_dequant_matmul(qw, cb, cb, 512, input);
    mlx::core::eval(w_gpu);

    // Time CPU dequant only (no matmul)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; i++) {
        auto r = dequantize_weight_cpu(qw, cb, cb, 512);
        mlx::core::eval(r);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5;

    // Time GPU fused dequant + matmul
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        auto r = fused_dequant_matmul(qw, cb, cb, 512, input);
        mlx::core::eval(r);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;

    double speedup = cpu_ms / gpu_ms;
    printf("  CPU dequant only:           %.1f ms\n", cpu_ms);
    printf("  GPU fused (dequant+matmul): %.1f ms\n", gpu_ms);
    printf("  Speedup: %.1fx\n", speedup);

    // GPU fused does MORE work (dequant + matmul) but should still be faster
    // than CPU dequant alone. A speedup > 3x confirms Metal kernel dispatch.
    assert(speedup > 3.0 && "GPU should be at least 3x faster than CPU — Metal not dispatching?");
    printf("  PASS: GPU fused kernel confirmed active (%.1fx faster)\n", speedup);
}

static void test_gpu_memory_used() {
    size_t peak = mlx::core::get_peak_memory();
    printf("  GPU peak memory: %zu MB\n", peak / (1024 * 1024));
    assert(peak > 0 && "GPU should have allocated memory during inference");
    printf("  PASS: GPU memory allocated\n");
}

int main() {
    printf("test_gpu_active:\n");
    test_gpu_available();
    test_fused_kernel_faster_than_cpu();
    test_gpu_memory_used();
    printf("All GPU activity tests passed.\n");
    return 0;
}
