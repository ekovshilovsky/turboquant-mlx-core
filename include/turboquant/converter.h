#pragma once

#include <mlx/mlx.h>
#include "quantizer.h"
#include <string>
#include <functional>

namespace turboquant {

/// Progress callback: (current_layer, total_layers, layer_name).
using ConversionProgressFn = std::function<void(int, int, const std::string&)>;

/// Configuration for model conversion.
struct ConversionConfig {
    QuantizerConfig quantizer;                   ///< Per-weight quantization settings
    std::string input_path;                      ///< Path to source HuggingFace model directory
    std::string output_path;                     ///< Path for converted model output
    ConversionProgressFn progress_callback;      ///< Optional progress reporting
    bool per_layer_codebooks = false;            ///< Fit Lloyd-Max codebooks to each layer's distribution
};

/// Convert a HuggingFace model to TurboQuant format.
/// Walks the model graph, identifies nn.Linear weights, quantizes each,
/// and writes output safetensors with TQ metadata.
/// Returns true on success.
bool convert_model(const ConversionConfig& config);

/// Validate that a converted model has correct TQ metadata and tensor shapes.
bool validate_converted_model(const std::string& model_path);

} // namespace turboquant
