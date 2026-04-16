#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>

namespace turboquant {

/// Information about a single node in the compute cluster.
struct NodeInfo {
    std::string hostname;       ///< Node hostname or IP
    int device_count;           ///< Number of GPU devices on this node
    size_t memory_bytes;        ///< Total unified memory in bytes
    float link_latency_us;      ///< Measured link latency in microseconds
};

/// Parallelism strategy for distributed inference.
enum class ShardStrategy {
    PipelineParallel,   ///< Split layers across nodes (best for high-latency links)
    TensorParallel,     ///< Shard weights across nodes (best for RDMA/Thunderbolt)
    Auto                ///< Auto-select based on measured link latency
};

/// Describes how a model is sharded across the cluster.
struct ShardPlan {
    ShardStrategy strategy;
    struct Assignment {
        int rank;
        int layer_start;        ///< First layer assigned (pipeline) or -1 (tensor)
        int layer_end;          ///< Last layer assigned (pipeline) or -1 (tensor)
        int head_start;         ///< First head assigned (tensor) or -1 (pipeline)
        int head_end;           ///< Last head assigned (tensor) or -1 (pipeline)
    };
    std::vector<Assignment> assignments;
};

/// Coordinator for multi-Mac distributed inference.
/// Manages cluster topology, shard planning, and activation transfer.
class TQDistributedCoordinator {
public:
    /// Initialize from hostfile. Backend: "jaccl", "ring", or "auto".
    bool init(const std::string& hostfile_path, const std::string& backend = "auto");

    /// Initialize for single-node inference (no hostfile needed).
    bool init_local();

    /// Analyze model + cluster topology to produce shard assignments.
    ShardPlan plan(int num_layers, int num_heads, int head_dim);

    /// Execute distributed forward pass.
    mlx::core::array forward(const mlx::core::array& input);

    int rank() const;
    int world_size() const;
    bool is_coordinator() const;

    /// Parse hostfile JSON. Returns list of nodes.
    static std::vector<NodeInfo> parse_hostfile(const std::string& path);

private:
    int rank_ = 0;
    int world_size_ = 1;
    ShardPlan plan_;
    std::string backend_;
    std::vector<NodeInfo> nodes_; ///< Cluster topology loaded from the hostfile.
};

} // namespace turboquant
