#include "turboquant_c/turboquant_c.h"
#include <assert.h>
#include <stdio.h>

static void test_create_free_lifecycle(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    tq_cluster_free(cluster);
    tq_cluster_free(NULL);
    printf("  PASS: create/free lifecycle\n");
}

static void test_invalid_create_params(void) {
    // Every dimension must be strictly positive. Zero and negative values
    // are rejected independently so the caller gets NULL rather than a
    // handle that cannot plan meaningful assignments.
    assert(tq_cluster_create(0, 28, 128) == NULL);
    assert(tq_cluster_create(-1, 28, 128) == NULL);
    assert(tq_cluster_create(64, 0, 128) == NULL);
    assert(tq_cluster_create(64, -1, 128) == NULL);
    assert(tq_cluster_create(64, 28, 0) == NULL);
    assert(tq_cluster_create(64, 28, -1) == NULL);
    printf("  PASS: invalid dimensions return NULL\n");
}

static void test_add_node_and_count(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_node_count(cluster) == 0);

    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);
    assert(tq_cluster_node_count(cluster) == 1);

    assert(tq_cluster_add_node(cluster, "host-b.local", 64.0) == 0);
    assert(tq_cluster_node_count(cluster) == 2);

    assert(tq_cluster_add_node(cluster, "host-c.local", 32.0) == 0);
    assert(tq_cluster_node_count(cluster) == 3);

    tq_cluster_free(cluster);
    printf("  PASS: add_node increments node_count\n");
}

static void test_add_node_rejects_invalid(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);

    assert(tq_cluster_add_node(NULL, "host.local", 64.0) == -1);
    assert(tq_cluster_add_node(cluster, NULL, 64.0) == -1);
    assert(tq_cluster_add_node(cluster, "host.local", 0.0) == -1);
    assert(tq_cluster_add_node(cluster, "host.local", -16.0) == -1);

    // All rejected calls must leave the node list untouched.
    assert(tq_cluster_node_count(cluster) == 0);

    tq_cluster_free(cluster);
    printf("  PASS: add_node rejects invalid inputs\n");
}

static void test_plan_assigns_all_layers(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-b.local", 64.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-c.local", 32.0) == 0);

    assert(tq_cluster_plan(cluster) == 0);

    // First rank starts at layer 0; last rank ends at num_layers; each
    // adjacent pair is contiguous with no gap or overlap.
    assert(tq_cluster_get_layer_start(cluster, 0) == 0);
    assert(tq_cluster_get_layer_end(cluster, 2) == 64);
    assert(tq_cluster_get_layer_end(cluster, 0) == tq_cluster_get_layer_start(cluster, 1));
    assert(tq_cluster_get_layer_end(cluster, 1) == tq_cluster_get_layer_start(cluster, 2));

    tq_cluster_free(cluster);
    printf("  PASS: plan covers all layers contiguously\n");
}

static void test_plan_memory_proportionality(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-b.local", 64.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-c.local", 32.0) == 0);

    assert(tq_cluster_plan(cluster) == 0);

    int span_0 = tq_cluster_get_layer_end(cluster, 0) - tq_cluster_get_layer_start(cluster, 0);
    int span_1 = tq_cluster_get_layer_end(cluster, 1) - tq_cluster_get_layer_start(cluster, 1);
    int span_2 = tq_cluster_get_layer_end(cluster, 2) - tq_cluster_get_layer_start(cluster, 2);

    // Memory-proportional assignment: the 128 GB node must receive more
    // layers than the 64 GB node, which in turn must receive more than
    // the 32 GB node. This invariant does not pin specific values;
    // it only enforces the monotonic contract of plan_memory_aware.
    assert(span_0 > span_1);
    assert(span_1 > span_2);

    tq_cluster_free(cluster);
    printf("  PASS: layer spans follow memory proportions\n");
}

static void test_query_before_plan_returns_neg_one(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);

    assert(tq_cluster_get_layer_start(cluster, 0) == -1);
    assert(tq_cluster_get_layer_end(cluster, 0) == -1);

    tq_cluster_free(cluster);
    printf("  PASS: query before plan returns -1\n");
}

static void test_plan_twice_is_error(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-b.local", 64.0) == 0);

    assert(tq_cluster_plan(cluster) == 0);
    assert(tq_cluster_plan(cluster) == -1);

    tq_cluster_free(cluster);
    printf("  PASS: second plan call returns -1\n");
}

static void test_get_layer_out_of_range(void) {
    tq_cluster_t cluster = tq_cluster_create(64, 28, 128);
    assert(cluster != NULL);
    assert(tq_cluster_add_node(cluster, "host-a.local", 128.0) == 0);
    assert(tq_cluster_add_node(cluster, "host-b.local", 64.0) == 0);
    assert(tq_cluster_plan(cluster) == 0);

    assert(tq_cluster_get_layer_start(cluster, 2) == -1);
    assert(tq_cluster_get_layer_end(cluster, 2) == -1);
    assert(tq_cluster_get_layer_start(cluster, -1) == -1);
    assert(tq_cluster_get_layer_end(cluster, -1) == -1);

    tq_cluster_free(cluster);
    printf("  PASS: out-of-range rank queries return -1\n");
}

static void test_free_null_safe(void) {
    tq_cluster_free(NULL);
    printf("  PASS: tq_cluster_free(NULL) is safe\n");
}

int main(void) {
    printf("test_c_api_cluster:\n");
    test_create_free_lifecycle();
    test_invalid_create_params();
    test_add_node_and_count();
    test_add_node_rejects_invalid();
    test_plan_assigns_all_layers();
    test_plan_memory_proportionality();
    test_query_before_plan_returns_neg_one();
    test_plan_twice_is_error();
    test_get_layer_out_of_range();
    test_free_null_safe();
    printf("All C API cluster tests passed.\n");
    return 0;
}
