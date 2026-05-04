#include "turboquant/converter.h"
#include "turboquant/turboquant.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static void print_usage() {
    printf("Usage: tq-convert --model <path> [--bits 4] [--residual-bits 4] [--output <path>]\n");
    printf("\n");
    printf("Convert a HuggingFace model to TurboQuant format.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --model <path>            Path to source model directory\n");
    printf("  --bits <n>                Primary quantization bits (default: 4)\n");
    printf("  --residual-bits <n>       Residual quantization bits (default: 4, 0 to disable)\n");
    printf("  --block-size <n>          WHT block size upper bound (default: 512)\n");
    printf("  --target-world-size <n>   Largest tensor-parallel world size the snapshot\n");
    printf("                            must support; constrains per-layer block sizes so\n");
    printf("                            in_features is divisible by world_size * block_size\n");
    printf("                            (default: 2, matching the common 2-Mac cluster).\n");
    printf("                            Use 1 for max-quality single-Mac snapshots; 4 or\n");
    printf("                            higher for larger clusters.\n");
    printf("  --sensitive-layers <n>    Keep first and last N transformer layers at fp16 (default: 0)\n");
    printf("  --per-layer-codebooks     Fit Lloyd-Max codebooks to each layer's distribution\n");
    printf("  --output <path>           Output directory. If omitted, the path is derived\n");
    printf("                            from the source model basename with a -TQ8-TP{N}\n");
    printf("                            suffix and placed beside the source directory\n");
    printf("                            (e.g., Qwen2.5-Coder-3B -> Qwen2.5-Coder-3B-TQ8-TP2).\n");
    printf("  --version                 Print version and exit\n");
}

/// Derive the default output directory when --output is not supplied.
///
/// The convention mirrors AWQ's "<model>-AWQ-INT4-G128" naming: the source
/// basename is suffixed with -TQ8-TP{N} so operators can see the topology
/// capability of the snapshot in both the local directory name and any
/// downstream HuggingFace repo name. The TQ8 token is shorthand for
/// "TurboQuant 8-bit-equivalent" (4 primary + 4 residual bits) — it is
/// hardcoded for the v1 release because that is the only configuration
/// validated end-to-end. A future per-bit-width naming change should
/// derive the suffix from primary_bits + residual_bits.
static std::string derive_output_path(const std::string& model_path,
                                      int target_world_size) {
    fs::path source(model_path);
    // Strip a trailing slash so basename resolves to the directory name
    // rather than to an empty path component.
    if (source.has_filename()) {
        // already resolves cleanly
    } else {
        source = source.parent_path();
    }
    const std::string basename = source.filename().string();
    const std::string suffix = "-TQ8-TP" + std::to_string(target_world_size);
    fs::path parent = source.parent_path();
    if (parent.empty()) {
        parent = ".";
    }
    return (parent / (basename + suffix)).string();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string model_path;
    std::string output_path;
    int bits = 4;
    int residual_bits = 4;
    int block_size = 512;
    int sensitive_layers = 0;
    int target_world_size = 2;
    bool per_layer_codebooks = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--bits" && i + 1 < argc) {
            bits = std::atoi(argv[++i]);
        } else if (arg == "--residual-bits" && i + 1 < argc) {
            residual_bits = std::atoi(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
            block_size = std::atoi(argv[++i]);
        } else if (arg == "--target-world-size" && i + 1 < argc) {
            target_world_size = std::atoi(argv[++i]);
        } else if (arg == "--sensitive-layers" && i + 1 < argc) {
            sensitive_layers = std::atoi(argv[++i]);
        } else if (arg == "--per-layer-codebooks") {
            per_layer_codebooks = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--version") {
            printf("tq-convert %s\n", turboquant::version());
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage();
        return 1;
    }

    if (target_world_size < 1) {
        fprintf(stderr, "Error: --target-world-size must be >= 1\n");
        return 1;
    }

    if (output_path.empty()) {
        output_path = derive_output_path(model_path, target_world_size);
    }

    turboquant::ConversionConfig config;
    config.input_path = model_path;
    config.output_path = output_path;
    config.quantizer.primary_bits = static_cast<uint8_t>(bits);
    config.quantizer.residual_bits = static_cast<uint8_t>(residual_bits);
    config.quantizer.block_size = static_cast<uint32_t>(block_size);
    config.quantizer.sensitive_layers_start = sensitive_layers;
    config.quantizer.sensitive_layers_end = sensitive_layers;
    config.quantizer.max_world_size = static_cast<uint32_t>(target_world_size);
    config.per_layer_codebooks = per_layer_codebooks;
    config.progress_callback = [](int current, int total, const std::string& name) {
        printf("[%d/%d] Converting %s\n", current, total, name.c_str());
    };

    printf("Converting: %s -> %s (TQ%d+%d, target_world_size=%d)\n",
           model_path.c_str(), output_path.c_str(), bits, residual_bits, target_world_size);

    if (!turboquant::convert_model(config)) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }

    printf("Conversion complete. Validating...\n");
    if (!turboquant::validate_converted_model(output_path)) {
        fprintf(stderr, "Warning: validation failed\n");
        return 1;
    }

    printf("Done.\n");
    return 0;
}
