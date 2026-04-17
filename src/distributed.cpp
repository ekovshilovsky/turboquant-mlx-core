#include "turboquant/distributed.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

namespace turboquant {

// ---------------------------------------------------------------------------
// Minimal JSON parser for the hostfile format:
//   { "nodes": [ {"hostname": "...", "device_count": N, "memory_bytes": N} ], "backend": "..." }
//
// Rather than introducing a third-party JSON dependency, this parser handles
// only the flat structure produced by the hostfile schema. It is not a
// general-purpose parser.
// ---------------------------------------------------------------------------

namespace {

/// Advance past whitespace characters in the source string starting at pos.
static void skip_whitespace(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
}

/// Extract an unescaped string value that begins immediately after pos points
/// at the opening double-quote. Advances pos past the closing double-quote.
/// Returns the extracted string, or empty string on parse failure.
static std::string read_string_value(const std::string& s, size_t& pos) {
    // pos must point at the opening '"'
    if (pos >= s.size() || s[pos] != '"') return {};
    ++pos; // skip opening quote
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos; // skip escape character
        }
        result += s[pos++];
    }
    if (pos < s.size()) ++pos; // skip closing quote
    return result;
}

/// Extract a decimal integer value starting at pos. Advances pos past the
/// last digit. Returns 0 on failure.
static long long read_integer_value(const std::string& s, size_t& pos) {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos == start) return 0;
    return std::stoll(s.substr(start, pos - start));
}

/// Locate the next occurrence of a JSON key (e.g. "hostname") in s starting
/// at pos and advance pos to the start of the corresponding value token.
/// Returns false if the key is not found before end of string.
static bool seek_key(const std::string& s, size_t& pos, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t found = s.find(needle, pos);
    if (found == std::string::npos) return false;
    pos = found + needle.size();
    skip_whitespace(s, pos);
    if (pos < s.size() && s[pos] == ':') ++pos; // skip ':'
    skip_whitespace(s, pos);
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// TQDistributedCoordinator — public interface
// ---------------------------------------------------------------------------

std::vector<NodeInfo> TQDistributedCoordinator::parse_hostfile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};

    std::ostringstream buf;
    buf << file.rdbuf();
    const std::string content = buf.str();

    // Locate the "nodes" array.
    size_t pos = 0;
    if (!seek_key(content, pos, "nodes")) return {};

    // Expect '[' to open the array.
    if (pos >= content.size() || content[pos] != '[') return {};
    ++pos;

    std::vector<NodeInfo> nodes;

    // Iterate over object entries within the array until the closing ']'.
    while (pos < content.size()) {
        skip_whitespace(content, pos);
        if (pos >= content.size()) break;

        // End of array.
        if (content[pos] == ']') break;

        // Start of an object entry.
        if (content[pos] != '{') {
            // Skip commas and any unexpected characters between objects.
            ++pos;
            continue;
        }

        // Find the closing brace for this object to scope key searches.
        size_t obj_end = content.find('}', pos);
        if (obj_end == std::string::npos) return {}; // malformed JSON

        std::string obj_str = content.substr(pos, obj_end - pos + 1);
        pos = obj_end + 1;

        // Extract required fields from the object substring.
        size_t field_pos = 0;

        if (!seek_key(obj_str, field_pos, "hostname")) return {};
        std::string hostname = read_string_value(obj_str, field_pos);
        if (hostname.empty()) return {};

        field_pos = 0;
        if (!seek_key(obj_str, field_pos, "device_count")) return {};
        int device_count = static_cast<int>(read_integer_value(obj_str, field_pos));

        field_pos = 0;
        if (!seek_key(obj_str, field_pos, "memory_bytes")) return {};
        size_t memory_bytes = static_cast<size_t>(read_integer_value(obj_str, field_pos));

        NodeInfo node;
        node.hostname      = std::move(hostname);
        node.device_count  = device_count;
        node.memory_bytes  = memory_bytes;
        node.link_latency_us = 0.0f; // Latency is measured at runtime; 0 until probed.

        nodes.push_back(std::move(node));
    }

    return nodes;
}

bool TQDistributedCoordinator::init(const std::string& hostfile_path,
                                    const std::string& backend) {
    std::vector<NodeInfo> nodes = parse_hostfile(hostfile_path);
    if (nodes.empty()) return false;

    nodes_     = std::move(nodes);
    rank_      = 0;                    // Coordinator is always rank 0 in the planning phase.
    world_size_ = static_cast<int>(nodes_.size());
    backend_   = backend;
    return true;
}

bool TQDistributedCoordinator::init_local() {
    rank_       = 0;
    world_size_ = 1;
    backend_    = "local";
    return true;
}

ShardPlan TQDistributedCoordinator::plan(int num_layers, int num_heads, int /*head_dim*/) {
    ShardPlan result;

    if (world_size_ == 1) {
        // Single-node deployment: assign the entire model to rank 0.
        result.strategy = ShardStrategy::PipelineParallel;
        result.assignments.push_back({0, 0, num_layers, 0, num_heads});
        return result;
    }

    // Auto strategy selection. When link latency has been measured we use it
    // to pick TensorParallel for sub-50 µs links (Thunderbolt / JACCL) and
    // PipelineParallel for higher-latency Ethernet links. Without an active
    // measurement pass all latencies default to 0, so we fall through to the
    // conservative Ethernet-safe default of PipelineParallel.
    ShardStrategy effective_strategy = ShardStrategy::PipelineParallel;
    if (!nodes_.empty()) {
        float total_latency = 0.0f;
        for (const auto& n : nodes_) total_latency += n.link_latency_us;
        float avg_latency = total_latency / static_cast<float>(nodes_.size());
        // Only upgrade to TensorParallel when latency is genuinely measured
        // and below the RDMA threshold.
        if (avg_latency > 0.0f && avg_latency < 50.0f) {
            effective_strategy = ShardStrategy::TensorParallel;
        }
    }
    result.strategy = effective_strategy;

    if (effective_strategy == ShardStrategy::PipelineParallel) {
        // Distribute layers as evenly as possible across ranks.
        for (int r = 0; r < world_size_; ++r) {
            int layer_start = r       * num_layers / world_size_;
            int layer_end   = (r + 1) * num_layers / world_size_;
            result.assignments.push_back({r, layer_start, layer_end, -1, -1});
        }
    } else {
        // TensorParallel: shard attention heads across ranks.
        for (int r = 0; r < world_size_; ++r) {
            int head_start = r       * num_heads / world_size_;
            int head_end   = (r + 1) * num_heads / world_size_;
            result.assignments.push_back({r, -1, -1, head_start, head_end});
        }
    }

    return result;
}

ShardPlan TQDistributedCoordinator::plan_memory_aware(
    int num_layers, int /*num_heads*/, int /*head_dim*/,
    const std::vector<NodeMemoryInfo>& nodes) {

    ShardPlan result;
    result.strategy = ShardStrategy::PipelineParallel;

    if (nodes.empty() || num_layers <= 0) return result;
    if (nodes.size() == 1) {
        result.assignments.push_back({0, 0, num_layers, -1, -1});
        return result;
    }

    // Sum usable memory across the cluster. Clamp the denominator so a
    // degenerate all-zero input still produces a valid plan instead of NaN.
    double total_mem = 0.0;
    for (const auto& n : nodes) total_mem += n.usable_memory_gb;
    if (total_mem <= 0.0) total_mem = 1.0;

    // Assign layers proportional to usable memory. The final node receives
    // whatever remains so the total is always exactly num_layers.
    int assigned = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        int rank = static_cast<int>(i);
        int layers_for_node;
        if (i + 1 == nodes.size()) {
            layers_for_node = num_layers - assigned;
        } else {
            layers_for_node = static_cast<int>(
                std::round(static_cast<double>(num_layers) * nodes[i].usable_memory_gb / total_mem));
            if (layers_for_node < 1) layers_for_node = 1;
            if (assigned + layers_for_node > num_layers) {
                layers_for_node = num_layers - assigned;
            }
        }
        result.assignments.push_back({rank, assigned, assigned + layers_for_node, -1, -1});
        assigned += layers_for_node;
    }

    return result;
}

mlx::core::array TQDistributedCoordinator::forward(const mlx::core::array& input) {
    if (world_size_ == 1) {
        // Single-node: pass activations through without transformation.
        return input;
    }

    // Multi-node activation transfer requires an active JACCL or Ring
    // collective session launched via `mpirun` or the JACCL process launcher.
    // Until that session is established, return input unchanged so that
    // coordinator-process planning logic can be exercised in isolation.
    return input;
}

int  TQDistributedCoordinator::rank()           const { return rank_; }
int  TQDistributedCoordinator::world_size()     const { return world_size_; }
bool TQDistributedCoordinator::is_coordinator() const { return rank_ == 0; }

} // namespace turboquant
