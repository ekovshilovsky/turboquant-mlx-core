// Node state machine tests. Validates transition rules, the LOW_MEMORY overlay,
// and queries used by the distributed coordinator to build the active set.

#include "turboquant/node_state.h"
#include <cassert>
#include <cstdio>

using namespace turboquant;

static void test_valid_transitions() {
    NodeStateManager mgr;
    auto id = mgr.add_node("m5-max", 128.0);

    assert(mgr.state(id) == NodeStateCode::Discovered);
    assert(mgr.transition(id, NodeStateCode::Evaluating));
    assert(mgr.transition(id, NodeStateCode::Syncing));
    assert(mgr.transition(id, NodeStateCode::Loading));
    assert(mgr.transition(id, NodeStateCode::Ready));
    assert(mgr.transition(id, NodeStateCode::Active));
    assert(mgr.state(id) == NodeStateCode::Active);
    printf("  PASS: valid transitions\n");
}

static void test_invalid_transition_rejected() {
    NodeStateManager mgr;
    auto id = mgr.add_node("m1-pro", 32.0);

    // Discovered cannot jump directly to Active.
    assert(!mgr.transition(id, NodeStateCode::Active));
    assert(mgr.state(id) == NodeStateCode::Discovered);
    printf("  PASS: invalid transition rejected\n");
}

static void test_skip_syncing_when_cached() {
    NodeStateManager mgr;
    auto id = mgr.add_node("m4-pro", 36.0);

    assert(mgr.transition(id, NodeStateCode::Evaluating));
    // Model already cached: Evaluating can skip straight to Loading.
    assert(mgr.transition(id, NodeStateCode::Loading));
    assert(mgr.state(id) == NodeStateCode::Loading);
    printf("  PASS: skip syncing when cached\n");
}

static void test_rejection_from_evaluating() {
    NodeStateManager mgr;
    auto id = mgr.add_node("tiny-mac", 4.0);

    assert(mgr.transition(id, NodeStateCode::Evaluating));
    assert(mgr.transition(id, NodeStateCode::Rejected));
    assert(mgr.state(id) == NodeStateCode::Rejected);
    printf("  PASS: rejection from evaluating\n");
}

static void test_disconnect_from_active() {
    NodeStateManager mgr;
    auto id = mgr.add_node("m5-max", 128.0);

    mgr.transition(id, NodeStateCode::Evaluating);
    mgr.transition(id, NodeStateCode::Loading);
    mgr.transition(id, NodeStateCode::Ready);
    mgr.transition(id, NodeStateCode::Active);

    assert(mgr.transition(id, NodeStateCode::Disconnected));
    assert(mgr.transition(id, NodeStateCode::Draining));
    assert(mgr.transition(id, NodeStateCode::Removed));
    assert(mgr.state(id) == NodeStateCode::Removed);
    printf("  PASS: disconnect from active\n");
}

static void test_low_memory_overlay() {
    NodeStateManager mgr;
    auto id = mgr.add_node("m1-pro", 32.0);

    mgr.transition(id, NodeStateCode::Evaluating);
    mgr.transition(id, NodeStateCode::Loading);
    mgr.transition(id, NodeStateCode::Ready);
    mgr.transition(id, NodeStateCode::Active);

    assert(!mgr.is_low_memory(id));
    mgr.set_low_memory(id, true);
    assert(mgr.is_low_memory(id));
    assert(mgr.state(id) == NodeStateCode::Active); // overlay does not change state
    mgr.set_low_memory(id, false);
    assert(!mgr.is_low_memory(id));
    printf("  PASS: low memory overlay\n");
}

static void test_active_node_list() {
    NodeStateManager mgr;
    auto a = mgr.add_node("node-a", 128.0);
    auto b = mgr.add_node("node-b", 64.0);
    (void)mgr.add_node("node-c", 32.0); // stays at Discovered

    for (auto id : {a, b}) {
        mgr.transition(id, NodeStateCode::Evaluating);
        mgr.transition(id, NodeStateCode::Loading);
        mgr.transition(id, NodeStateCode::Ready);
        mgr.transition(id, NodeStateCode::Active);
    }

    auto active = mgr.active_nodes();
    assert(active.size() == 2);
    printf("  PASS: active node list\n");
}

int main() {
    printf("test_node_state:\n");
    test_valid_transitions();
    test_invalid_transition_rejected();
    test_skip_syncing_when_cached();
    test_rejection_from_evaluating();
    test_disconnect_from_active();
    test_low_memory_overlay();
    test_active_node_list();
    printf("All node state tests passed.\n");
    return 0;
}
