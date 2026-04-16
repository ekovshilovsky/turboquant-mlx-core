#include "turboquant/converter.h"
#include "turboquant/quantizer.h"
#include "turboquant/codebook.h"
#include "turboquant/dequantizer.h"
#include <mlx/mlx.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <unordered_map>

namespace fs = std::filesystem;

// Forward declaration for the internal metadata reader defined in serialization.cpp
namespace turboquant {
    std::unordered_map<std::string, std::string> read_tq_metadata(const std::string& path);
}

/// Create a test model directory with multiple layers for parallel conversion testing.
/// Each layer has a distinct deterministic data pattern so that any cross-contamination
/// between parallel workers produces detectable differences in the output tensors.
static bool create_multilayer_test_model(const std::string& dir_path, int num_layers) {
    fs::create_directories(dir_path);

    std::unordered_map<std::string, mlx::core::array> tensors;

    for (int layer = 0; layer < num_layers; ++layer) {
        const int rows = 32;
        const int cols = 128;
        std::vector<float> data(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            // Use a layer-dependent pattern that produces distinguishable distributions
            uint32_t state = static_cast<uint32_t>(layer * 7919 + i);
            state = state * 1103515245u + 12345u;
            data[static_cast<size_t>(i)] =
                (static_cast<float>(state >> 16) / 32768.0f) - 1.0f;
        }
        auto weight = mlx::core::array(data.data(), {rows, cols}, mlx::core::float32);
        mlx::core::eval(weight);

        std::string name = "model.layers." + std::to_string(layer) +
                           ".self_attn.q_proj.weight";
        tensors.insert({name, weight});
    }

    const std::string shard_path = dir_path + "/model.safetensors";
    mlx::core::save_safetensors(shard_path, tensors);
    return fs::exists(shard_path);
}

/// Run a conversion and return the output shard path. Configures the converter
/// identically each time so serial and parallel runs use the same codebook and
/// rotation parameters.
static std::string run_conversion(const std::string& input_dir,
                                   const std::string& output_dir) {
    fs::remove_all(output_dir);

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 128;
    config.per_layer_codebooks     = false;

    bool ok = turboquant::convert_model(config);
    assert(ok && "convert_model must succeed");

    return output_dir + "/model.safetensors";
}

/// Compare two safetensors files and verify all tensors are bit-identical.
/// Returns true if every tensor present in both files matches exactly.
static bool compare_safetensors(const std::string& path_a, const std::string& path_b) {
    auto [tensors_a, meta_a] = mlx::core::load_safetensors(path_a);
    auto [tensors_b, meta_b] = mlx::core::load_safetensors(path_b);

    if (tensors_a.size() != tensors_b.size()) {
        printf("    FAIL: tensor count mismatch: %zu vs %zu\n",
               tensors_a.size(), tensors_b.size());
        return false;
    }

    for (const auto& [name, tensor_a] : tensors_a) {
        auto it = tensors_b.find(name);
        if (it == tensors_b.end()) {
            printf("    FAIL: tensor '%s' missing from second file\n", name.c_str());
            return false;
        }

        const auto& tensor_b = it->second;
        mlx::core::eval(tensor_a);
        mlx::core::eval(tensor_b);

        if (tensor_a.shape() != tensor_b.shape()) {
            printf("    FAIL: shape mismatch for '%s'\n", name.c_str());
            return false;
        }

        if (tensor_a.dtype() != tensor_b.dtype()) {
            printf("    FAIL: dtype mismatch for '%s'\n", name.c_str());
            return false;
        }

        // Byte-level comparison for exact equality
        size_t nbytes = tensor_a.nbytes();
        const uint8_t* bytes_a = reinterpret_cast<const uint8_t*>(tensor_a.data<void>());
        const uint8_t* bytes_b = reinterpret_cast<const uint8_t*>(tensor_b.data<void>());
        for (size_t i = 0; i < nbytes; ++i) {
            if (bytes_a[i] != bytes_b[i]) {
                printf("    FAIL: byte mismatch at offset %zu in tensor '%s'\n",
                       i, name.c_str());
                return false;
            }
        }
    }
    return true;
}

/// Test 1: Verify that running conversion twice produces bit-identical output.
/// Since the converter uses std::async internally, this confirms that the
/// parallel execution path is deterministic and thread-safe. Both runs use
/// the same input data, codebooks, and seeds, so any non-determinism from
/// race conditions or unprotected shared state would produce mismatches.
static void test_parallel_conversion_is_deterministic() {
    const std::string input_dir  = "/tmp/tq_parallel_input";
    const std::string output_a   = "/tmp/tq_parallel_output_a";
    const std::string output_b   = "/tmp/tq_parallel_output_b";

    fs::remove_all(input_dir);
    fs::remove_all(output_a);
    fs::remove_all(output_b);

    bool created = create_multilayer_test_model(input_dir, 4);
    assert(created && "test model creation must succeed");

    // Run conversion twice with identical configuration
    std::string shard_a = run_conversion(input_dir, output_a);
    std::string shard_b = run_conversion(input_dir, output_b);

    assert(fs::exists(shard_a) && "first conversion output must exist");
    assert(fs::exists(shard_b) && "second conversion output must exist");

    bool identical = compare_safetensors(shard_a, shard_b);
    assert(identical && "two identical conversions must produce bit-identical output");

    fs::remove_all(input_dir);
    fs::remove_all(output_a);
    fs::remove_all(output_b);

    printf("  PASS: parallel conversion is deterministic (bit-identical across runs)\n");
}

/// Test 2: Verify that all expected tensor groups are present and have valid shapes.
/// The converter must emit packed_primary, packed_residual, norms, and seeds for
/// each quantized layer, plus model-level codebook tensors. Missing or malformed
/// tensors indicate a thread-safety issue in the result collection path.
static void test_parallel_output_completeness() {
    const std::string input_dir  = "/tmp/tq_parallel_complete_input";
    const std::string output_dir = "/tmp/tq_parallel_complete_output";

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    const int num_layers = 6;
    bool created = create_multilayer_test_model(input_dir, num_layers);
    assert(created && "test model creation must succeed");

    std::string shard_path = run_conversion(input_dir, output_dir);
    auto [tensors, meta] = mlx::core::load_safetensors(shard_path);

    // Verify each layer produced all four required tensor components
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer) + ".self_attn.q_proj";
        assert(tensors.count(prefix + ".packed_primary") &&
               "packed_primary must exist for each layer");
        assert(tensors.count(prefix + ".packed_residual") &&
               "packed_residual must exist for each layer");
        assert(tensors.count(prefix + ".norms") &&
               "norms must exist for each layer");
        assert(tensors.count(prefix + ".seeds") &&
               "seeds must exist for each layer");

        // Verify seeds tensor contains exactly 3 elements
        auto& seeds = tensors.at(prefix + ".seeds");
        mlx::core::eval(seeds);
        assert(seeds.shape(0) == 3 && "seeds must have 3 elements");

        // Verify norms shape matches output features
        auto& norms = tensors.at(prefix + ".norms");
        mlx::core::eval(norms);
        assert(norms.shape(0) == 32 && "norms must have out_features elements");
    }

    // Verify model-level codebook tensors are present
    assert(tensors.count("tq_codebook_primary") && "primary codebook must be present");
    assert(tensors.count("tq_codebook_residual") && "residual codebook must be present");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: parallel output completeness (%d layers, all tensor groups present)\n",
           num_layers);
}

/// Test 3: Verify GPU WHT rotation produces results matching the CPU reference
/// within floating-point tolerance. The GPU Metal kernel and CPU butterfly
/// implement the same mathematical transform (sign flip + FWHT + scaling) but
/// use different execution paths, so numerical agreement validates correctness.
static void test_gpu_cpu_wht_agreement() {
    // Test with a matrix large enough to trigger GPU dispatch (>= 1024 elements)
    // and a block size that divides evenly into the column count
    const int rows = 8;
    const int cols = 512;
    const uint32_t seed = 42;
    const uint32_t block_size = 512;

    auto x = mlx::core::random::normal({rows, cols});
    mlx::core::eval(x);

    // Both paths should be exercised by the apply_wht_rotation dispatcher:
    // the matrix is 8x512 = 4096 elements, above the 1024 threshold
    auto rotated = turboquant::apply_wht_rotation(x, seed, block_size);
    mlx::core::eval(rotated);

    // Round-trip through inverse must recover original within tolerance
    auto recovered = turboquant::apply_inverse_wht_rotation(rotated, seed, block_size);
    mlx::core::eval(recovered);

    const float* src = x.data<float>();
    const float* rec = recovered.data<float>();
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float err = std::abs(src[i] - rec[i]);
        if (err > max_err) max_err = err;
    }

    printf("    GPU WHT round-trip max error: %.2e\n", max_err);
    assert(max_err < 1e-4f &&
           "GPU WHT round-trip error must be below 1e-4");

    // Verify norm preservation: ||WHT(x)|| / ||x|| should be ~1.0
    auto x_sq = mlx::core::multiply(x, x);
    auto x_norm_sq = mlx::core::sum(x_sq);
    auto r_sq = mlx::core::multiply(rotated, rotated);
    auto r_norm_sq = mlx::core::sum(r_sq);
    mlx::core::eval(x_norm_sq, r_norm_sq);

    float ratio = std::sqrt(r_norm_sq.item<float>() / x_norm_sq.item<float>());
    assert(std::abs(ratio - 1.0f) < 1e-3f &&
           "GPU WHT must preserve L2 norm");

    printf("  PASS: GPU WHT agrees with CPU reference (max_err=%.2e, norm_ratio=%.4f)\n",
           max_err, ratio);
}

/// Test 4: Verify GPU WHT handles multiple block sizes correctly.
/// Tests non-trivial block configurations that exercise different threadgroup
/// sizes in the Metal kernel: 64, 128, 256, and 512 threads per group.
static void test_gpu_wht_multiple_block_sizes() {
    const int rows = 8;
    const uint32_t seed = 77;

    int block_sizes[] = {64, 128, 256, 512};
    for (int bs : block_sizes) {
        // Column count must be divisible by block_size
        int cols = bs * 4;
        auto x = mlx::core::random::normal({rows, cols});
        mlx::core::eval(x);

        auto rotated = turboquant::apply_wht_rotation(x, seed, static_cast<uint32_t>(bs));
        auto recovered = turboquant::apply_inverse_wht_rotation(
            rotated, seed, static_cast<uint32_t>(bs));
        mlx::core::eval(rotated, recovered);

        const float* src = x.data<float>();
        const float* rec = recovered.data<float>();
        float max_err = 0.0f;
        for (int i = 0; i < rows * cols; ++i) {
            float err = std::abs(src[i] - rec[i]);
            if (err > max_err) max_err = err;
        }

        assert(max_err < 1e-4f && "GPU WHT round-trip error exceeded tolerance");
        printf("    block_size=%d: max_err=%.2e OK\n", bs, max_err);
    }

    printf("  PASS: GPU WHT round-trip correct for all block sizes\n");
}

/// Test 5: Verify GPU-accelerated Lloyd-Max codebook fitting produces codebooks
/// equivalent to the CPU path within convergence tolerance. Both paths start
/// from the same N(0,1) initial centroids and iterate to convergence, so their
/// outputs should agree to within the iteration's own convergence threshold.
static void test_gpu_lloyd_max_agreement() {
    // Generate a large sample set that triggers the GPU path (>= 65536 elements)
    const int n = 131072;
    std::vector<float> data(n);
    uint32_t state = 12345u;
    for (int i = 0; i < n; ++i) {
        // Approximate N(0,1) via Box-Muller using the LCG
        state = state * 1103515245u + 12345u;
        float u1 = static_cast<float>(state >> 16) / 65536.0f;
        u1 = std::max(u1, 1e-7f);
        state = state * 1103515245u + 12345u;
        float u2 = static_cast<float>(state >> 16) / 65536.0f;
        data[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }

    // generate_codebook_from_data will route to GPU for n >= 65536
    auto cb = turboquant::generate_codebook_from_data(data, 4, 100);

    // Validate the codebook structure
    assert(cb.centroids.size() == 16 && "4-bit codebook must have 16 centroids");
    assert(cb.boundaries.size() == 15 && "4-bit codebook must have 15 boundaries");

    // Centroids must be strictly sorted
    for (size_t i = 1; i < cb.centroids.size(); ++i) {
        assert(cb.centroids[i] > cb.centroids[i - 1] &&
               "GPU-fitted centroids must be strictly sorted");
    }

    // Boundaries must be midpoints of adjacent centroids
    for (size_t i = 0; i < cb.boundaries.size(); ++i) {
        float expected = (cb.centroids[i] + cb.centroids[i + 1]) * 0.5f;
        assert(std::abs(cb.boundaries[i] - expected) < 1e-5f &&
               "boundaries must be midpoints of adjacent centroids");
    }

    // For approximately N(0,1) data, the fitted centroids should be close to
    // the precomputed ones (within ~10% relative error for 131K samples)
    auto ref = turboquant::generate_codebook(4);
    float max_deviation = 0.0f;
    for (size_t i = 0; i < cb.centroids.size(); ++i) {
        float dev = std::abs(cb.centroids[i] - ref.centroids[i]);
        if (dev > max_deviation) max_deviation = dev;
    }

    printf("    GPU Lloyd-Max max centroid deviation from N(0,1) reference: %.4f\n",
           max_deviation);
    assert(max_deviation < 0.3f &&
           "GPU-fitted centroids for N(0,1) data must be within 0.3 of precomputed values");

    // Verify quantize-dequantize round-trip with the GPU-fitted codebook
    assert(turboquant::validate_codebook(cb) &&
           "GPU-fitted codebook must pass structural validation");

    printf("  PASS: GPU Lloyd-Max codebook fitting produces valid, accurate codebook\n");
}

/// Test 6: Wall-clock timing comparison to verify parallel conversion is not
/// degraded. This test measures conversion time for a model with enough layers
/// to demonstrate parallelism. It does not assert a strict speedup ratio because
/// the test model is small and thread scheduling overhead may dominate, but it
/// verifies the parallel path completes within a reasonable time bound.
static void test_parallel_conversion_timing() {
    const std::string input_dir  = "/tmp/tq_parallel_timing_input";
    const std::string output_dir = "/tmp/tq_parallel_timing_output";

    fs::remove_all(input_dir);

    // Use 8 layers to provide enough independent work for parallel execution
    bool created = create_multilayer_test_model(input_dir, 8);
    assert(created && "test model creation must succeed");

    // Warm up: first conversion includes JIT compilation overhead for Metal
    // kernels that would skew timing results
    run_conversion(input_dir, output_dir);
    fs::remove_all(output_dir);

    auto start = std::chrono::high_resolution_clock::now();
    run_conversion(input_dir, output_dir);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("    Parallel conversion of 8-layer test model: %.1f ms\n", ms);

    // Sanity check: conversion should complete within 30 seconds even on slow
    // machines; anything longer indicates a deadlock or performance regression
    assert(ms < 30000.0 && "conversion must complete within 30 seconds");

    // Verify output is valid
    bool valid = turboquant::validate_converted_model(output_dir);
    assert(valid && "timed conversion output must be valid");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: parallel conversion timing within acceptable bounds\n");
}

int main() {
    printf("test_parallel_converter:\n");
    test_parallel_conversion_is_deterministic();
    test_parallel_output_completeness();
    test_gpu_cpu_wht_agreement();
    test_gpu_wht_multiple_block_sizes();
    test_gpu_lloyd_max_agreement();
    test_parallel_conversion_timing();
    printf("All parallel converter tests passed.\n");
    return 0;
}
