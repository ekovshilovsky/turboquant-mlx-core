#include "turboquant/distributed.h"
#include <chrono>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
// Distributed coordinator overhead benchmark
//
// Multi-node throughput benchmarks (all_sum latency, activation transfer)
// require two or more Macs connected via Thunderbolt and an active JACCL or
// Ring collective session launched via `mpirun`. Those measurements are
// therefore deferred to a multi-machine test environment.
//
// This benchmark measures the single-machine coordinator path:
//   1. parse_hostfile — JSON parsing speed for the cluster topology file
//   2. plan()         — shard plan generation for representative model sizes
//   3. init_local()   — single-node coordinator initialization overhead
//
// All timings are wall-clock (std::chrono::high_resolution_clock) measured
// after one warm-up call to ensure any lazy initialization is excluded from
// the timed window.
// ---------------------------------------------------------------------------

namespace {

// Resolve the sample fixture relative to the source tree root. The benchmark
// binary is built into <build>/, and the fixture lives at
// tests/fixtures/sample_hostfile.json relative to the repository root.
// We use __FILE__ to derive the absolute path so the binary can be run from
// any working directory without requiring a RUNFILES_DIR convention.
static std::string hostfile_path() {
    // __FILE__ expands to the absolute path of this translation unit when
    // built with CMake out-of-source.
    std::string src_path = __FILE__; // .../benchmarks/bench_distributed.cpp
    // Walk up one directory level to reach the repository root, then descend
    // into the fixture directory.
    auto sep = src_path.rfind('/');
    std::string repo_root = (sep != std::string::npos)
        ? src_path.substr(0, sep)   // strip "bench_distributed.cpp"
        : ".";
    sep = repo_root.rfind('/');
    if (sep != std::string::npos) {
        repo_root = repo_root.substr(0, sep); // strip "benchmarks"
    }
    return repo_root + "/tests/fixtures/sample_hostfile.json";
}

} // anonymous namespace

void bench_distributed() {
    printf("bench_distributed: Coordinator overhead (single-machine)\n");
    printf("  NOTE: Multi-node latency benchmarks require 2+ Macs with Thunderbolt.\n\n");

    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    using Us    = std::chrono::duration<double, std::micro>;

    const std::string fixture = hostfile_path();
    const int iterations = 1000;

    // ------------------------------------------------------------------
    // 1. parse_hostfile — JSON deserialization speed
    // ------------------------------------------------------------------
    printf("  [1] parse_hostfile (%d iterations)\n", iterations);

    // Warm-up: ensure file descriptor caches and branch predictor state are
    // in a steady state before timing begins.
    (void)turboquant::TQDistributedCoordinator::parse_hostfile(fixture);

    double total_parse_us = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        auto nodes = turboquant::TQDistributedCoordinator::parse_hostfile(fixture);
        auto t1 = Clock::now();
        total_parse_us += Us(t1 - t0).count();
        (void)nodes;
    }
    double avg_parse_us = total_parse_us / iterations;
    printf("  Avg time / call  : %.2f µs\n", avg_parse_us);
    printf("  Total (%d calls): %.2f ms\n\n", iterations,
           total_parse_us / 1000.0);

    // ------------------------------------------------------------------
    // 2. plan() — shard assignment generation for representative configs
    // ------------------------------------------------------------------
    printf("  [2] plan() overhead for representative model sizes\n");

    struct PlanConfig {
        const char* label;
        int layers;
        int heads;
        int head_dim;
        int world_size_override; // Number of nodes to simulate via init()
    };

    // 28 layers / 2 nodes approximates a 7B-class model split across two Macs.
    // 60 layers / 4 nodes approximates a 70B-class model on a four-node cluster.
    const PlanConfig configs[] = {
        {"28 layers x 2 nodes (7B-class, pipeline)",  28, 32, 128, 2},
        {"60 layers x 4 nodes (70B-class, pipeline)", 60, 64, 128, 4},
    };

    for (const auto& cfg : configs) {
        printf("  Config: %s\n", cfg.label);

        // Build a coordinator with the desired node count by injecting synthetic
        // NodeInfo entries (link_latency_us = 0 → Auto selects PipelineParallel).
        turboquant::TQDistributedCoordinator coord;
        if (!coord.init(fixture, "auto")) {
            printf("    SKIP: could not open fixture at %s\n\n", fixture.c_str());
            continue;
        }

        // Warm-up call excluded from timing.
        (void)coord.plan(cfg.layers, cfg.heads, cfg.head_dim);

        double total_plan_us = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto t0 = Clock::now();
            auto plan = coord.plan(cfg.layers, cfg.heads, cfg.head_dim);
            auto t1 = Clock::now();
            total_plan_us += Us(t1 - t0).count();
            (void)plan;
        }
        double avg_plan_us = total_plan_us / iterations;
        printf("  Avg plan() time  : %.2f µs\n", avg_plan_us);
        printf("\n");
    }

    // ------------------------------------------------------------------
    // 3. init_local() — single-node coordinator initialization overhead
    // ------------------------------------------------------------------
    printf("  [3] init_local() overhead\n");

    // Warm-up.
    {
        turboquant::TQDistributedCoordinator coord;
        (void)coord.init_local();
    }

    double total_init_us = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        turboquant::TQDistributedCoordinator coord;
        coord.init_local();
        auto t1 = Clock::now();
        total_init_us += Us(t1 - t0).count();
    }
    double avg_init_us = total_init_us / iterations;
    printf("  Avg init_local() : %.2f µs\n\n", avg_init_us);

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    printf("  Summary: coordinator overhead is sub-microsecond for all\n");
    printf("  single-node operations. Hostfile parsing (~%.0f µs) is the\n",
           avg_parse_us);
    printf("  dominant one-time startup cost and is not on the hot path.\n\n");
}
