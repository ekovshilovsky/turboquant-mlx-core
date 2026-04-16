#include "turboquant/converter.h"
#include "turboquant/turboquant.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static void print_usage() {
    printf("Usage: tq-convert --model <path> [--bits 4] [--residual-bits 4] [--output <path>]\n");
    printf("\n");
    printf("Convert a HuggingFace model to TurboQuant format.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --model <path>         Path to source model directory\n");
    printf("  --bits <n>             Primary quantization bits (default: 4)\n");
    printf("  --residual-bits <n>    Residual quantization bits (default: 4, 0 to disable)\n");
    printf("  --block-size <n>       WHT block size (default: 512)\n");
    printf("  --sensitive-layers <n> Keep first and last N transformer layers at fp16 (default: 0)\n");
    printf("  --per-layer-codebooks  Fit Lloyd-Max codebooks to each layer's distribution\n");
    printf("  --output <path>        Output directory (default: <model>-tq<bits>)\n");
    printf("  --version              Print version and exit\n");
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

    if (output_path.empty()) {
        output_path = model_path + "-tq" + std::to_string(bits);
    }

    turboquant::ConversionConfig config;
    config.input_path = model_path;
    config.output_path = output_path;
    config.quantizer.primary_bits = static_cast<uint8_t>(bits);
    config.quantizer.residual_bits = static_cast<uint8_t>(residual_bits);
    config.quantizer.block_size = static_cast<uint32_t>(block_size);
    config.quantizer.sensitive_layers_start = sensitive_layers;
    config.quantizer.sensitive_layers_end = sensitive_layers;
    config.per_layer_codebooks = per_layer_codebooks;
    config.progress_callback = [](int current, int total, const std::string& name) {
        printf("[%d/%d] Converting %s\n", current, total, name.c_str());
    };

    printf("Converting: %s -> %s (TQ%d+%d)\n",
           model_path.c_str(), output_path.c_str(), bits, residual_bits);

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
