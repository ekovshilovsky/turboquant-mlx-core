#include "turboquant/linear.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <mlx/io.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

/// Real model path for end-to-end benchmarking. Set TQ_MODEL_PATH environment
/// variable to point to a converted TQ8 model directory, or leave unset to
/// use the built-in test fixture.
static const char* kDefaultModelPath = nullptr;

/// Fallback fixture path used when no real model is available.
static const char* kFixturePath =
    "tests/fixtures/tiny_model_tq8";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true if `key` ends with the given `suffix`.
static bool ends_with(const std::string& key, const std::string& suffix) {
    return key.size() >= suffix.size() &&
           key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Scan an array-map from a loaded safetensors file and return the layer
/// prefix of the first weight tensor group found (i.e. first key that ends
/// in ".packed_primary").  Returns empty string if none found.
static std::string find_first_layer_prefix(
    const std::unordered_map<std::string, mlx::core::array>& tensors)
{
    std::string suffix = ".packed_primary";
    // Collect all matching keys so the selection is deterministic
    std::vector<std::string> candidates;
    for (const auto& kv : tensors) {
        if (ends_with(kv.first, suffix)) {
            candidates.push_back(kv.first);
        }
    }
    if (candidates.empty()) {
        return "";
    }
    std::sort(candidates.begin(), candidates.end());
    // Strip the trailing ".packed_primary" to obtain the bare layer prefix
    const std::string& first = candidates.front();
    return first.substr(0, first.size() - suffix.size());
}

// ---------------------------------------------------------------------------
// Synthetic fixture: build a small QuantizedWeight without file I/O
// ---------------------------------------------------------------------------

/// Construct a minimal QuantizedWeight for the tiny_model fixture.
/// Dimensions mirror the fixture config: hidden_size=64, intermediate=128.
/// We use a 64-column weight (64 half-columns packed into 32 uint8 bytes).
static turboquant::QuantizedWeight make_fixture_weight(
    int out_features,
    int in_features,
    const turboquant::Codebook& primary_cb,
    const turboquant::Codebook& residual_cb,
    const turboquant::QuantizerConfig& cfg)
{
    std::mt19937 rng(0xF1F0U);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(in_features)));

    std::vector<float> weight_data(static_cast<size_t>(out_features) * in_features);
    for (auto& v : weight_data) {
        v = dist(rng);
    }
    auto weight = mlx::core::array(
        weight_data.data(), {out_features, in_features}, mlx::core::float32);
    mlx::core::eval(weight);

    return turboquant::quantize_weight(weight, primary_cb, residual_cb, cfg);
}

// ---------------------------------------------------------------------------
// Main benchmark body
// ---------------------------------------------------------------------------

void bench_e2e() {
    printf("bench_e2e: Full model layer tok/s (end-to-end forward pass)\n");

    // -----------------------------------------------------------------------
    // Locate a safetensors shard from a real converted model.
    // The benchmark accepts an optional model directory via the environment
    // variable TQ_MODEL_PATH:
    //   TQ_MODEL_PATH=/path/to/model ./build/tq_benchmarks
    // When not set, the hardcoded Coder-3B path is tried first.  If neither
    // shard is found on disk, the benchmark synthesises a weight matrix from
    // the tiny_model_tq8 fixture dimensions and benchmarks that instead.
    // -----------------------------------------------------------------------

    const char* env_path = std::getenv("TQ_MODEL_PATH");
    std::string model_dir = env_path ? env_path
                          : kDefaultModelPath ? kDefaultModelPath
                          : kFixturePath;

    // Build codebooks and quantizer config — shared by both the real-model
    // and the fixture paths.  shared_rotation=true (the default) stores
    // residual quantization in the primary WHT domain, allowing the fused
    // kernel to execute a single butterfly pass per block at inference time.
    turboquant::QuantizerConfig cfg;
    cfg.primary_bits    = 4;
    cfg.residual_bits   = 4;
    cfg.block_size      = 512;
    cfg.shared_rotation = true;
    cfg.norm_correction = true;

    auto primary_cb  = turboquant::generate_codebook(static_cast<uint8_t>(cfg.primary_bits));
    auto residual_cb = turboquant::generate_codebook(static_cast<uint8_t>(cfg.residual_bits));

    // The converter writes multi-shard models as model-NNNNN-of-MMMMM.safetensors
    // and single-shard models as model.safetensors.
    std::string shard_multi  = model_dir + "/model-00001-of-00002.safetensors";
    std::string shard_single = model_dir + "/model.safetensors";

    // QuantizedWeight has no default constructor (mlx::core::array members are
    // non-default-constructible), so we use std::optional and emplace after all
    // four constituent tensors are confirmed to be present.
    std::optional<turboquant::QuantizedWeight> qw_opt;
    int in_features       = 0;
    int out_features      = 0;
    bool loaded_from_file = false;

    for (const std::string* path : { &shard_multi, &shard_single }) {
        FILE* probe = std::fopen(path->c_str(), "rb");
        if (!probe) {
            continue;
        }
        std::fclose(probe);

        printf("  Loading shard : %s\n", path->c_str());

        // Load all tensors and file-level metadata from the safetensors shard.
        // SafetensorsLoad = pair<unordered_map<string, array>, unordered_map<string, string>>
        auto [tensors, metadata] = mlx::core::load_safetensors(*path);

        std::string prefix = find_first_layer_prefix(tensors);
        if (prefix.empty()) {
            printf("  Warning: no TQ weight tensors found in shard, skipping.\n");
            continue;
        }

        printf("  Using layer   : %s\n", prefix.c_str());

        auto it_packed   = tensors.find(prefix + ".packed_primary");
        auto it_residual = tensors.find(prefix + ".packed_residual");
        auto it_norms    = tensors.find(prefix + ".norms");
        auto it_seeds    = tensors.find(prefix + ".seeds");

        if (it_packed == tensors.end() || it_residual == tensors.end() ||
            it_norms  == tensors.end() || it_seeds   == tensors.end()) {
            printf("  Warning: incomplete tensor group for '%s', skipping.\n",
                   prefix.c_str());
            continue;
        }

        // packed_primary shape: [out_features, in_features/2] (4-bit nibble packing).
        // Two 4-bit indices are packed into each uint8, so the true column count
        // is twice the stored column dimension.
        const auto& packed = it_packed->second;
        out_features = packed.shape(0);
        in_features  = packed.shape(1) * 2;

        qw_opt.emplace(turboquant::QuantizedWeight{
            it_packed->second,
            it_residual->second,
            it_norms->second,
            it_seeds->second
        });

        // Materialise all tensors on the GPU before timing begins
        mlx::core::eval(qw_opt->packed_primary);
        mlx::core::eval(qw_opt->packed_residual);
        mlx::core::eval(qw_opt->norms);
        mlx::core::eval(qw_opt->seeds);

        loaded_from_file = true;
        break;
    }

    if (!loaded_from_file) {
        // Fallback: synthesise a QuantizedWeight from tiny_model_tq8 dimensions.
        // hidden_size=64, intermediate_size=128 → benchmark a 128x64 weight layer.
        printf("  Falling back to tiny_model_tq8 fixture (128 x 64)\n");
        out_features = 128;
        in_features  = 64;
        qw_opt.emplace(
            make_fixture_weight(out_features, in_features, primary_cb, residual_cb, cfg));
    }

    const turboquant::QuantizedWeight& qw = *qw_opt;

    printf("  Weight dims   : %d x %d  (out x in)\n", out_features, in_features);

    // -----------------------------------------------------------------------
    // Construct TurboQuantLinear and a single-token input vector
    // -----------------------------------------------------------------------
    turboquant::TurboQuantLinear tq_linear(
        in_features, out_features, qw, primary_cb, residual_cb,
        static_cast<uint32_t>(cfg.block_size));

    std::vector<float> input_data(static_cast<size_t>(in_features), 1.0f);
    auto input = mlx::core::array(
        input_data.data(), {1, in_features}, mlx::core::float32);
    mlx::core::eval(input);

    // -----------------------------------------------------------------------
    // Warm-up: one forward pass to JIT-compile Metal kernels and populate
    // GPU caches before the timed loop begins.
    // -----------------------------------------------------------------------
    {
        auto warmup_out = tq_linear.forward(input);
        mlx::core::eval(warmup_out);
    }

    // -----------------------------------------------------------------------
    // Timed benchmark: 100 synchronous forward passes, batch=1.
    // mlx::core::eval() flushes the lazy computation graph and blocks until
    // the GPU has completed all pending work, ensuring accurate wall-clock
    // measurements that capture the full dispatch-to-completion latency.
    // -----------------------------------------------------------------------
    const int iterations = 100;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto t0  = std::chrono::high_resolution_clock::now();
        auto out = tq_linear.forward(input);
        mlx::core::eval(out);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double ms_per_iter = total_ms / iterations;
    // Each forward pass processes exactly one token (batch=1)
    double toks_per_s  = 1000.0 / ms_per_iter;

    printf("\n");
    printf("  Iterations    : %d\n", iterations);
    printf("  Total time    : %.2f ms\n", total_ms);
    printf("  Time / token  : %.3f ms\n", ms_per_iter);
    printf("  Throughput    : %.1f tok/s\n", toks_per_s);
    if (loaded_from_file) {
        printf("  Source        : real model  (%s)\n", model_dir.c_str());
    } else {
        printf("  Source        : fixture     (%s)\n", kFixturePath);
    }
    printf("\n");
}
