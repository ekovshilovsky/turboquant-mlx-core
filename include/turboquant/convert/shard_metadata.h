#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace turboquant::convert {

/// Describes a single tensor within a converted TurboQuant model, including
/// the information a distributed weight loader needs to shard or replicate
/// the tensor across tensor-parallel ranks without re-parsing safetensors
/// internals. Populated by the convert tool at write time and consumed by
/// SwiftLM's shard-aware loader.
struct TensorEntry {
    std::string name;              ///< Fully qualified tensor name (e.g. model.layers.0.self_attn.q_proj.weight)
    std::vector<int64_t> shape;    ///< Logical shape in elements (not bytes)
    std::string dtype;             ///< Storage dtype tag (e.g. "tq8", "f16", "f32")
    std::string file;              ///< Safetensors shard filename containing the payload
    int64_t byte_offset = 0;       ///< Absolute byte offset of the payload within the shard
    int64_t byte_length = 0;       ///< Payload length in bytes
    int32_t shard_axis = -1;       ///< Axis to shard along; -1 indicates a replicated tensor
    std::string shard_strategy;    ///< One of: "column_parallel", "row_parallel", "replicated", "expert_parallel"
    std::string codebook_key;      ///< Companion codebook tensor name; empty when the tensor is not TQ-coded
    std::string rotation_key;      ///< Companion rotation tensor name; empty when the tensor is not TQ-coded
    int32_t expert_index = -1;     ///< MoE expert index; -1 when the tensor is not an expert weight
};

/// Accumulates tensor entries and model-level attributes, then serializes the
/// collection as the sidecar tq_shard_metadata.json document.
///
/// The emitted JSON is intentionally small (no tensor payloads, only pointers
/// and shapes), so holding the full document in memory is never a concern even
/// for large models.
class ShardMetadata {
public:
    void add_tensor(TensorEntry entry);

    void set_architecture(std::string arch);
    void set_hidden_size(int64_t h);
    void set_num_attention_heads(int32_t n);
    void set_intermediate_size(int64_t i);
    void set_num_experts(int32_t n);
    void set_top_k(int32_t k);

    /// Serialize the accumulated metadata as a pretty-printed JSON document.
    /// Stable key ordering keeps diffs reviewable and comparisons deterministic.
    std::string to_json_string() const;

    /// Schema version. Bump whenever the JSON structure changes in a way that
    /// would require readers to update; additive optional fields do not bump it.
    static constexpr int FORMAT_VERSION = 1;

private:
    std::string architecture_;
    int64_t hidden_size_ = 0;
    int32_t num_attention_heads_ = 0;
    int64_t intermediate_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 0;
    std::vector<TensorEntry> tensors_;
};

}  // namespace turboquant::convert
