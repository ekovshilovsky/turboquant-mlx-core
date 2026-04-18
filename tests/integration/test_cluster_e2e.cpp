// Multi-thread cluster simulation. Exercises the distributed runtime's
// transport, node state machine, and memory-aware shard planner together in a
// single process using loopback TCP. A coordinator thread accepts two worker
// threads, collects heartbeats, plans a 3-node pipeline, then simulates the
// loss of one worker and verifies the planner redistributes all layers across
// the surviving nodes.

#include "turboquant/distributed.h"
#include "turboquant/node_state.h"
#include "turboquant/transport.h"

#include <cassert>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

using namespace turboquant;

namespace {

// Model shape used for planning. Values are representative of a mid-size
// transformer and are unrelated to the transport mechanics under test; they
// exist so the planner has concrete arguments to work against.
constexpr int kNumLayers = 64;
constexpr int kNumHeads  = 28;
constexpr int kHeadDim   = 128;

// Per-node usable memory in GB. The coordinator counts as rank 0.
constexpr double kCoordMemGb   = 128.0;
constexpr double kWorker1MemGb = 64.0;
constexpr double kWorker2MemGb = 32.0;

// Drive a worker's lifecycle through Evaluating, Loading, Ready, Active so
// subsequent disconnect tests have a node in the only state from which
// Disconnected is reachable.
void bring_node_active(NodeStateManager& mgr, NodeId id) {
    bool ok = true;
    ok &= mgr.transition(id, NodeStateCode::Evaluating);
    ok &= mgr.transition(id, NodeStateCode::Loading);
    ok &= mgr.transition(id, NodeStateCode::Ready);
    ok &= mgr.transition(id, NodeStateCode::Active);
    assert(ok);
}

// Run a single worker: connect to the coordinator, send one heartbeat with the
// supplied rank and available memory, then wait for the coordinator's "done"
// ACK before returning. The ACK handshake guarantees the worker thread does
// not exit (and close its socket) before the coordinator has finished reading
// its heartbeat, eliminating any dependence on sleeps for ordering.
void run_worker(std::shared_future<int> port_future,
                uint32_t rank,
                float available_memory_gb) {
    int port = port_future.get();
    assert(port > 0);

    TcpChannel ch;
    bool connected = ch.connect("127.0.0.1", port);
    assert(connected);

    Heartbeat hb;
    hb.rank = rank;
    hb.state = NodeStateCode::Active;
    hb.available_memory_gb = available_memory_gb;
    hb.layer_start = -1;
    hb.layer_end = -1;
    hb.tokens_processed = 0;
    hb.avg_layer_ms = 0.0f;
    hb.syncing_percent = -1;
    bool sent = ch.send_heartbeat(hb);
    assert(sent);

    // Block until the coordinator signals teardown. This is the synchronization
    // primitive for worker-exit ordering; the worker must not tear down its
    // channel until the coordinator has completed its reads and planning work.
    bool done = ch.recv_ack();
    assert(done);
}

// Helper: assert that a shard plan is a contiguous, complete cover of the
// layer range [0, num_layers). This is the invariant the memory-aware planner
// is contractually required to produce regardless of node memory weighting.
void assert_plan_covers_all_layers(const ShardPlan& plan, int num_layers,
                                   size_t expected_assignments) {
    assert(plan.strategy == ShardStrategy::PipelineParallel);
    assert(plan.assignments.size() == expected_assignments);

    int expected_next_start = 0;
    for (size_t i = 0; i < plan.assignments.size(); ++i) {
        const auto& a = plan.assignments[i];
        assert(a.rank == static_cast<int>(i));
        assert(a.layer_start == expected_next_start);
        assert(a.layer_end > a.layer_start);
        // Pipeline-parallel assignments carry no head slice.
        assert(a.head_start == -1);
        assert(a.head_end == -1);
        expected_next_start = a.layer_end;
    }
    assert(expected_next_start == num_layers);
}

void test_three_node_join() {
    TcpListener listener;
    int port = listener.bind_any("127.0.0.1");
    assert(port > 0);
    listener.listen(4);

    // Publish the bound port to both worker threads. A promise/future is used
    // instead of a sleep so workers attempt their connect only after the
    // listener is guaranteed to be ready.
    std::promise<int> port_promise;
    std::shared_future<int> port_future = port_promise.get_future().share();

    std::thread worker1(run_worker, port_future, /*rank=*/1,
                        static_cast<float>(kWorker1MemGb));
    std::thread worker2(run_worker, port_future, /*rank=*/2,
                        static_cast<float>(kWorker2MemGb));
    port_promise.set_value(port);

    NodeStateManager mgr;

    // Accept both workers. The accept() calls block until each worker's
    // connect() completes, so the order in which the two worker threads
    // actually hit the listener does not affect correctness.
    TcpChannel ch1 = listener.accept();
    TcpChannel ch2 = listener.accept();

    NodeId worker1_id = mgr.add_node("w1", kWorker1MemGb);
    NodeId worker2_id = mgr.add_node("w2", kWorker2MemGb);
    bring_node_active(mgr, worker1_id);
    bring_node_active(mgr, worker2_id);
    assert(mgr.state(worker1_id) == NodeStateCode::Active);
    assert(mgr.state(worker2_id) == NodeStateCode::Active);
    assert(mgr.active_nodes().size() == 2);

    // Read one heartbeat from each worker. The first heartbeat received is not
    // guaranteed to be from worker 1 because the accept ordering depends on
    // kernel scheduling; dispatch on rank to make the assertions order-free.
    Heartbeat hb_a, hb_b;
    bool ok_a = ch1.recv_heartbeat(hb_a);
    bool ok_b = ch2.recv_heartbeat(hb_b);
    assert(ok_a);
    assert(ok_b);

    const Heartbeat& hb_w1 = (hb_a.rank == 1) ? hb_a : hb_b;
    const Heartbeat& hb_w2 = (hb_a.rank == 2) ? hb_a : hb_b;
    assert(hb_w1.rank == 1);
    assert(hb_w2.rank == 2);
    assert(hb_w1.state == NodeStateCode::Active);
    assert(hb_w2.state == NodeStateCode::Active);
    assert(hb_w1.available_memory_gb == static_cast<float>(kWorker1MemGb));
    assert(hb_w2.available_memory_gb == static_cast<float>(kWorker2MemGb));

    // Build a three-node plan: coordinator (rank 0) plus both workers.
    std::vector<NodeMemoryInfo> cluster = {
        {"coord", kCoordMemGb},
        {"w1",    kWorker1MemGb},
        {"w2",    kWorker2MemGb},
    };
    TQDistributedCoordinator coord;
    ShardPlan plan = coord.plan_memory_aware(kNumLayers, kNumHeads, kHeadDim, cluster);

    assert_plan_covers_all_layers(plan, kNumLayers, /*expected_assignments=*/3);

    // Memory-weighted proportionality: the coordinator has the most memory
    // and must own strictly more layers than either worker, and worker 1
    // (64 GB) must own strictly more than worker 2 (32 GB).
    int layers_coord = plan.assignments[0].layer_end - plan.assignments[0].layer_start;
    int layers_w1    = plan.assignments[1].layer_end - plan.assignments[1].layer_start;
    int layers_w2    = plan.assignments[2].layer_end - plan.assignments[2].layer_start;
    assert(layers_coord > layers_w1);
    assert(layers_w1 > layers_w2);

    // Release both workers so their threads can exit cleanly.
    bool s1 = ch1.send_ack();
    bool s2 = ch2.send_ack();
    assert(s1 && s2);
    worker1.join();
    worker2.join();
    printf("  PASS: three-node join, heartbeats, and initial shard plan\n");
}

void test_disconnect_triggers_redistribution() {
    TcpListener listener;
    int port = listener.bind_any("127.0.0.1");
    assert(port > 0);
    listener.listen(4);

    std::promise<int> port_promise;
    std::shared_future<int> port_future = port_promise.get_future().share();

    std::thread worker1(run_worker, port_future, /*rank=*/1,
                        static_cast<float>(kWorker1MemGb));
    std::thread worker2(run_worker, port_future, /*rank=*/2,
                        static_cast<float>(kWorker2MemGb));
    port_promise.set_value(port);

    NodeStateManager mgr;
    TcpChannel ch1 = listener.accept();
    TcpChannel ch2 = listener.accept();

    NodeId worker1_id = mgr.add_node("w1", kWorker1MemGb);
    NodeId worker2_id = mgr.add_node("w2", kWorker2MemGb);
    bring_node_active(mgr, worker1_id);
    bring_node_active(mgr, worker2_id);

    Heartbeat hb1, hb2;
    assert(ch1.recv_heartbeat(hb1));
    assert(ch2.recv_heartbeat(hb2));

    TQDistributedCoordinator coord;
    std::vector<NodeMemoryInfo> full_cluster = {
        {"coord", kCoordMemGb},
        {"w1",    kWorker1MemGb},
        {"w2",    kWorker2MemGb},
    };
    ShardPlan initial_plan = coord.plan_memory_aware(
        kNumLayers, kNumHeads, kHeadDim, full_cluster);
    assert_plan_covers_all_layers(initial_plan, kNumLayers, 3);

    // Simulate worker 2 going offline. Drive it through every remaining
    // lifecycle state to exercise the full teardown path in the state manager.
    assert(mgr.transition(worker2_id, NodeStateCode::Disconnected));
    assert(mgr.state(worker2_id) == NodeStateCode::Disconnected);
    assert(mgr.transition(worker2_id, NodeStateCode::Draining));
    assert(mgr.transition(worker2_id, NodeStateCode::Removed));
    assert(mgr.state(worker2_id) == NodeStateCode::Removed);

    // After worker 2 is removed, only the coordinator and worker 1 remain
    // eligible for shard assignment.
    std::vector<NodeMemoryInfo> surviving_cluster = {
        {"coord", kCoordMemGb},
        {"w1",    kWorker1MemGb},
    };
    ShardPlan redistributed = coord.plan_memory_aware(
        kNumLayers, kNumHeads, kHeadDim, surviving_cluster);
    assert_plan_covers_all_layers(redistributed, kNumLayers, /*expected_assignments=*/2);

    // Worker 1's share must grow when worker 2 is gone because all 64 layers
    // are now divided between only two nodes.
    int w1_layers_initial = initial_plan.assignments[1].layer_end
                          - initial_plan.assignments[1].layer_start;
    int w1_layers_after   = redistributed.assignments[1].layer_end
                          - redistributed.assignments[1].layer_start;
    assert(w1_layers_after > w1_layers_initial);

    bool s1 = ch1.send_ack();
    bool s2 = ch2.send_ack();
    assert(s1 && s2);
    worker1.join();
    worker2.join();
    printf("  PASS: disconnect triggers redistribution across surviving nodes\n");
}

} // namespace

int main() {
    printf("test_cluster_e2e:\n");
    test_three_node_join();
    test_disconnect_triggers_redistribution();
    printf("All cluster e2e tests passed.\n");
    return 0;
}
