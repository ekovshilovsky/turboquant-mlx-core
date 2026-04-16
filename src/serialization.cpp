#include "turboquant/converter.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <string>
#include <unordered_map>

namespace turboquant {

/// Read safetensors file and extract TQ metadata.
/// Returns empty map if the file contains no TQ metadata or cannot be read.
std::unordered_map<std::string, std::string> read_tq_metadata(const std::string& path) {
    auto [tensors, metadata] = mlx::core::load_safetensors(path);
    return metadata;
}

/// Write quantized tensors to safetensors with TQ metadata and codebook storage.
///
/// Tensor layout per named weight:
///   {name}.packed_primary   — uint8 [out, in/2]: nibble-packed primary indices
///   {name}.packed_residual  — uint8 [out, in/2]: nibble-packed residual indices
///   {name}.norms            — float32 [out]: corrected row norms
///   {name}.seeds            — uint32 [3]: {seed_primary, seed_residual, block_size}
///
/// Model-level codebook tensors (shared mode):
///   tq_codebook_primary     — float32 centroids for primary pass
///   tq_codebook_residual    — float32 centroids for residual pass
///
/// Per-layer codebook tensors (when per_layer_codebooks map is non-empty):
///   {name}.codebook_primary  — float32 layer-specific primary centroids
///   {name}.codebook_residual — float32 layer-specific residual centroids
bool write_tq_safetensors(
    const std::string& path,
    const std::unordered_map<std::string, QuantizedWeight>& weights,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    const std::unordered_map<std::string, std::pair<Codebook, Codebook>>& per_layer_codebooks) {

    std::unordered_map<std::string, mlx::core::array> tensors;

    // Serialize each quantized weight as four named tensors.
    // mlx::core::array has no default constructor, so we use insert rather than
    // operator[] to avoid the implicit default-construction that operator[] performs.
    for (const auto& [name, qw] : weights) {
        tensors.insert({name + ".packed_primary",  qw.packed_primary});
        tensors.insert({name + ".packed_residual", qw.packed_residual});
        tensors.insert({name + ".norms",           qw.norms});
        tensors.insert({name + ".seeds",           qw.seeds});
    }

    bool has_per_layer = !per_layer_codebooks.empty();

    if (has_per_layer) {
        // Store per-layer codebook centroids alongside each weight's tensors.
        // Adds ~128 bytes per layer (16 centroids x 4 bytes x 2 codebooks).
        for (const auto& [name, cb_pair] : per_layer_codebooks) {
            const auto& [layer_primary, layer_residual] = cb_pair;
            tensors.insert({name + ".codebook_primary", mlx::core::array(
                layer_primary.centroids.data(),
                {static_cast<int>(layer_primary.centroids.size())})});
            tensors.insert({name + ".codebook_residual", mlx::core::array(
                layer_residual.centroids.data(),
                {static_cast<int>(layer_residual.centroids.size())})});
        }
    }

    // Store model-level codebook centroids as fallback and for backward
    // compatibility with readers that do not support per-layer codebooks
    tensors.insert({"tq_codebook_primary", mlx::core::array(
        primary_codebook.centroids.data(),
        {static_cast<int>(primary_codebook.centroids.size())})});

    tensors.insert({"tq_codebook_residual", mlx::core::array(
        residual_codebook.centroids.data(),
        {static_cast<int>(residual_codebook.centroids.size())})});

    // Build metadata so downstream readers can identify and route TQ files
    std::unordered_map<std::string, std::string> metadata;
    metadata["quantization_method"] = "turboquant";
    metadata["tq_version"]          = "1";
    metadata["tq_primary_bits"]     = std::to_string(primary_codebook.bits);
    metadata["tq_residual_bits"]    = std::to_string(residual_codebook.bits);
    metadata["tq_total_bits"]       = std::to_string(
        primary_codebook.bits + residual_codebook.bits);
    if (has_per_layer) {
        metadata["tq_per_layer_codebooks"] = "true";
    }

    mlx::core::save_safetensors(path, tensors, metadata);
    return true;
}

} // namespace turboquant
