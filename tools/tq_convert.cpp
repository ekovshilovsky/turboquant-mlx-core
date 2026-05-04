#include "turboquant/converter.h"
#include "turboquant/turboquant.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static void print_usage() {
    printf("Usage: tq-convert --model <path> [--output <path>] [--draft]\n");
    printf("\n");
    printf("Convert a HuggingFace model to TurboQuant format. The default flags\n");
    printf("produce a release-quality snapshot suitable for serving; use --draft\n");
    printf("for fast smoke-test conversions during development.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --model <path>            Path to source model directory\n");
    printf("  --output <path>           Output directory. If omitted, the path is derived\n");
    printf("                            from the source model basename with a -TQ8-TP{N}\n");
    printf("                            suffix and placed beside the source directory\n");
    printf("                            (e.g., Qwen2.5-Coder-3B -> Qwen2.5-Coder-3B-TQ8-TP2).\n");
    printf("  --draft                   Fast conversion preset for development and CI:\n");
    printf("                            disables --per-layer-codebooks and sets\n");
    printf("                            --sensitive-layers 0. Roughly 30x faster but\n");
    printf("                            10-20x worse perplexity delta. Do not ship to\n");
    printf("                            users; intended for round-trip smoke tests.\n");
    printf("  --target-world-size <n>   Largest tensor-parallel world size the snapshot\n");
    printf("                            must support; constrains per-layer block sizes so\n");
    printf("                            in_features is divisible by world_size * block_size\n");
    printf("                            (default: 2, matching the common 2-Mac cluster).\n");
    printf("                            Use 1 for max-quality single-Mac snapshots; 4 or\n");
    printf("                            higher for larger clusters.\n");
    printf("\n");
    printf("Advanced overrides (defaults are validated for production use):\n");
    printf("  --bits <n>                Primary quantization bits (default: 4)\n");
    printf("  --residual-bits <n>       Residual quantization bits (default: 4, 0 to disable)\n");
    printf("  --block-size <n>          WHT block size upper bound (default: 512)\n");
    printf("  --sensitive-layers <n>    Keep first and last N transformer layers at fp16\n");
    printf("                            (default: 4). Validated for 32B/7B/3B and hybrid\n");
    printf("                            Qwen3.5 architectures; smaller models can use 2.\n");
    printf("  --no-per-layer-codebooks  Use a single global Lloyd-Max codebook instead of\n");
    printf("                            per-layer fits. Faster but increases perplexity\n");
    printf("                            delta by 10-20x; prefer --draft for quick runs.\n");
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
    // Production-quality defaults: 4 sensitive boundary layers and per-layer
    // Lloyd-Max codebooks. Validated to deliver <1% perplexity delta against
    // fp16 across Qwen2.5 (3B/7B/32B) and hybrid Qwen3.5 architectures. Users
    // who need fast iteration pass --draft to flip both off.
    int sensitive_layers = 4;
    int target_world_size = 2;
    bool per_layer_codebooks = true;
    bool draft = false;

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
            // Retained for backward compatibility; the flag is now the
            // default and the explicit form is a no-op.
            per_layer_codebooks = true;
        } else if (arg == "--no-per-layer-codebooks") {
            per_layer_codebooks = false;
        } else if (arg == "--draft") {
            draft = true;
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

    // --draft is a preset that flips the quality-affecting defaults off for
    // fast iteration. It is applied after argument parsing, so explicit
    // per-flag overrides on the same command line lose to --draft regardless
    // of order; that is intentional — the preset is the authoritative
    // statement of intent. Topology and bit-width settings are unaffected.
    if (draft) {
        sensitive_layers = 0;
        per_layer_codebooks = false;
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

    const char* quality_label = draft
        ? "draft"
        : (per_layer_codebooks && sensitive_layers > 0 ? "production" : "custom");
    printf("Converting: %s -> %s\n", model_path.c_str(), output_path.c_str());
    printf("  Quality: %s (sensitive_layers=%d, per_layer_codebooks=%s)\n",
           quality_label, sensitive_layers, per_layer_codebooks ? "yes" : "no");
    printf("  Format:  TQ%d+%d, target_world_size=%d\n",
           bits, residual_bits, target_world_size);

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
