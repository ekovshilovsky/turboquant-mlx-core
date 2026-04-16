#include "turboquant_c/turboquant_c.h"
#include <assert.h>
#include <stdio.h>

static void test_create_and_free(void) {
    tq_kv_cache_t cache = tq_kv_cache_create(2, 2, 64, 3, 1024, 256);
    assert(cache != NULL);
    tq_kv_cache_free(cache);
    printf("  PASS: create/free lifecycle\n");
}

static void test_free_null_safe(void) {
    tq_kv_cache_free(NULL);
    printf("  PASS: tq_kv_cache_free(NULL) is safe\n");
}

static void test_seq_length_initial(void) {
    tq_kv_cache_t cache = tq_kv_cache_create(2, 2, 64, 3, 1024, 256);
    assert(cache != NULL);
    int len = tq_kv_cache_seq_length(cache);
    assert(len == 0);
    tq_kv_cache_free(cache);
    printf("  PASS: initial seq_length is 0\n");
}

int main(void) {
    printf("test_c_api_kv_cache:\n");
    test_create_and_free();
    test_free_null_safe();
    test_seq_length_initial();
    printf("All C API KV cache tests passed.\n");
    return 0;
}
