#include "turboquant_c/turboquant_c.h"
#include <assert.h>
#include <stdio.h>

static void test_init_local(void) {
    tq_coordinator_t coord = tq_distributed_init_local();
    assert(coord != NULL);
    tq_distributed_free(coord);
    printf("  PASS: local init lifecycle\n");
}

static void test_single_node_rank(void) {
    tq_coordinator_t coord = tq_distributed_init_local();
    assert(coord != NULL);
    int rank = tq_distributed_rank(coord);
    assert(rank == 0);
    int ws = tq_distributed_world_size(coord);
    assert(ws == 1);
    tq_distributed_free(coord);
    printf("  PASS: single-node rank=0, world_size=1\n");
}

static void test_free_null_safe(void) {
    tq_distributed_free(NULL);
    printf("  PASS: tq_distributed_free(NULL) is safe\n");
}

static void test_init_invalid_hostfile(void) {
    tq_coordinator_t coord = tq_distributed_init("/nonexistent/hostfile.json", "auto");
    assert(coord == NULL);
    printf("  PASS: tq_distributed_init returns NULL for invalid hostfile\n");
}

int main(void) {
    printf("test_c_api_distributed:\n");
    test_init_local();
    test_single_node_rank();
    test_free_null_safe();
    test_init_invalid_hostfile();
    printf("All C API distributed tests passed.\n");
    return 0;
}
