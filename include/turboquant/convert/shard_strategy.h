#pragma once

#include <cstdint>
#include <string>

namespace turboquant::convert {

/// Infer the tensor-parallel shard strategy from a weight's fully qualified
/// name. Attention/MLP input projections shard along the output-feature axis
/// (column-parallel); attention/MLP output projections shard along the
/// input-feature axis (row-parallel). Everything else (embeddings, norms,
/// biases, lm_head) is treated as replicated.
///
/// Returns one of: "column_parallel", "row_parallel", "replicated".
/// Expert-parallel strategies are identified separately by the MoE-aware
/// caller and are not inferred from the name alone.
std::string infer_shard_strategy(const std::string& tensor_name);

/// Map a strategy tag to the canonical shard axis. Returns -1 for replicated
/// and expert-parallel tensors to signal "no axis" to the metadata writer.
int32_t infer_shard_axis(const std::string& strategy);

}  // namespace turboquant::convert
