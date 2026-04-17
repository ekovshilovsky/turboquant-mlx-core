#include "turboquant/node_state.h"

namespace turboquant {

NodeId NodeStateManager::add_node(const std::string& hostname, double physical_memory_gb) {
    NodeId id = next_id_++;
    ManagedNode node;
    node.id = id;
    node.hostname = hostname;
    node.physical_memory_gb = physical_memory_gb;
    node.available_memory_gb = physical_memory_gb;
    node.state = NodeStateCode::Discovered;
    node.low_memory = false;
    node.layer_start = -1;
    node.layer_end = -1;
    nodes_.emplace(id, std::move(node));
    return id;
}

bool NodeStateManager::is_valid_transition(NodeStateCode current, NodeStateCode new_state) {
    switch (current) {
        case NodeStateCode::Discovered:
            return new_state == NodeStateCode::Evaluating;
        case NodeStateCode::Evaluating:
            return new_state == NodeStateCode::Syncing
                || new_state == NodeStateCode::Loading   // skip sync if model already cached
                || new_state == NodeStateCode::Rejected;
        case NodeStateCode::Rejected:
            return false; // terminal
        case NodeStateCode::Syncing:
            return new_state == NodeStateCode::Loading
                || new_state == NodeStateCode::Rejected;
        case NodeStateCode::Loading:
            return new_state == NodeStateCode::Ready;
        case NodeStateCode::Ready:
            return new_state == NodeStateCode::Active;
        case NodeStateCode::Active:
            return new_state == NodeStateCode::Disconnected;
        case NodeStateCode::Disconnected:
            return new_state == NodeStateCode::Draining;
        case NodeStateCode::Draining:
            return new_state == NodeStateCode::Removed;
        case NodeStateCode::Removed:
            return false; // terminal
    }
    return false;
}

bool NodeStateManager::transition(NodeId id, NodeStateCode new_state) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;
    if (!is_valid_transition(it->second.state, new_state)) return false;
    it->second.state = new_state;
    // LOW_MEMORY is meaningful only while Active. Clear on any exit from Active
    // so a node cannot retain stale pressure flags across reassignment.
    if (new_state != NodeStateCode::Active) {
        it->second.low_memory = false;
    }
    return true;
}

NodeStateCode NodeStateManager::state(NodeId id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return NodeStateCode::Removed;
    return it->second.state;
}

void NodeStateManager::set_low_memory(NodeId id, bool low) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return;
    if (it->second.state == NodeStateCode::Active) {
        it->second.low_memory = low;
    }
}

bool NodeStateManager::is_low_memory(NodeId id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;
    return it->second.low_memory;
}

std::vector<NodeId> NodeStateManager::active_nodes() const {
    std::vector<NodeId> result;
    for (const auto& [id, node] : nodes_) {
        if (node.state == NodeStateCode::Active) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<NodeId> NodeStateManager::ready_nodes() const {
    std::vector<NodeId> result;
    for (const auto& [id, node] : nodes_) {
        if (node.state == NodeStateCode::Ready) {
            result.push_back(id);
        }
    }
    return result;
}

const ManagedNode* NodeStateManager::node(NodeId id) const {
    auto it = nodes_.find(id);
    return it != nodes_.end() ? &it->second : nullptr;
}

void NodeStateManager::set_available_memory(NodeId id, double gb) {
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        it->second.available_memory_gb = gb;
    }
}

void NodeStateManager::set_layers(NodeId id, int32_t start, int32_t end) {
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        it->second.layer_start = start;
        it->second.layer_end = end;
    }
}

} // namespace turboquant
