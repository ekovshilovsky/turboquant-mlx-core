#pragma once

#include <filesystem>
#include <string>

namespace turboquant::convert {

/// Result of a sidecar emission run. Populated by emit_sidecar_for_directory
/// so callers can surface a clear diagnostic when inputs are malformed without
/// relying on logs.
struct EmitSidecarResult {
    bool ok = false;               ///< Overall success flag
    std::string error;             ///< Human-readable error when ok is false
    std::filesystem::path sidecar_path;  ///< Written sidecar path when ok is true
    int tensor_count = 0;          ///< Number of logical tensor entries written
    int column_parallel_count = 0;
    int row_parallel_count = 0;
    int replicated_count = 0;
    int expert_parallel_count = 0;
};

/// Scan an already-TurboQuant-converted model directory and write
/// tq_shard_metadata.json alongside the safetensors files. Used by both
/// the tq-convert post-write path and the tq-emit-sidecar retrofit tool so
/// they produce byte-identical sidecars from the same on-disk layout.
///
/// Expects converted_dir to contain config.json plus at least one
/// *.safetensors shard. Model-level attributes (architecture, hidden_size,
/// num_attention_heads, intermediate_size, MoE expert counts) are read from
/// config.json; per-tensor entries are built by parsing each shard's
/// safetensors header. The TQ dtype tag is synthesized from the
/// quantization_config block that tq-convert injects into config.json.
///
/// Returns an EmitSidecarResult with ok=true and the full per-strategy
/// counts on success; ok=false with a populated error string on failure.
EmitSidecarResult emit_sidecar_for_directory(const std::filesystem::path& converted_dir);

}  // namespace turboquant::convert
