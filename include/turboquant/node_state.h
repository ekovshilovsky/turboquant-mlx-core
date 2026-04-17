#pragma once

// Node lifecycle state management for distributed TurboQuant clusters.
// Enforces valid state transitions, tracks the LOW_MEMORY overlay on active
// nodes, and exposes queries used by the coordinator to build the working set.

#include "turboquant/transport.h"  // for NodeStateCode

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace turboquant {

/// Unique identifier for a node in the cluster. Monotonically increasing
/// within a coordinator; never reused after removal.
using NodeId = uint32_t;

/// Extended node record tracked by the state manager. Fields in addition to
/// state are updated out-of-band via heartbeats and planner assignments.
struct ManagedNode {
    NodeId id = 0;
    std::string hostname;
    double physical_memory_gb = 0.0;
    double available_memory_gb = 0.0;
    NodeStateCode state = NodeStateCode::Discovered;
    bool low_memory = false;
    int32_t layer_start = -1;
    int32_t layer_end = -1;
};

/// Manages the lifecycle state of all nodes in the cluster. Enforces valid
/// state transitions; invalid transitions are rejected without side effects.
class NodeStateManager {
public:
    /// Add a new node in the Discovered state. Returns its unique ID.
    NodeId add_node(const std::string& hostname, double physical_memory_gb);

    /// Attempt a state transition. Returns false if the transition is not
    /// allowed from the current state; no side effects on rejection.
    bool transition(NodeId id, NodeStateCode new_state);

    /// Get the current state of a node. Returns Removed for unknown IDs.
    NodeStateCode state(NodeId id) const;

    /// Set or clear the LOW_MEMORY overlay. Only effective on Active nodes;
    /// cleared automatically on any state change away from Active.
    void set_low_memory(NodeId id, bool low);

    /// Check whether a node has the LOW_MEMORY flag set.
    bool is_low_memory(NodeId id) const;

    /// Return all nodes currently in the Active state.
    std::vector<NodeId> active_nodes() const;

    /// Return all nodes currently in the Ready state.
    std::vector<NodeId> ready_nodes() const;

    /// Retrieve the full record for a node, or nullptr if no such node.
    const ManagedNode* node(NodeId id) const;

    /// Record the node's currently available memory (reported via heartbeat).
    void set_available_memory(NodeId id, double gb);

    /// Record the pipeline layer range assigned to a node.
    void set_layers(NodeId id, int32_t start, int32_t end);

private:
    NodeId next_id_ = 0;
    std::unordered_map<NodeId, ManagedNode> nodes_;

    /// Validate whether a transition from current to new_state is allowed.
    static bool is_valid_transition(NodeStateCode current, NodeStateCode new_state);
};

} // namespace turboquant
