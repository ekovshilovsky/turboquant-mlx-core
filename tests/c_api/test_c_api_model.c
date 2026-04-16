#include "turboquant_c/turboquant_c.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_load_valid_model(void) {
    tq_model_t model = tq_model_load("tests/fixtures/tiny_model_tq8");
    assert(model != NULL);
    printf("  PASS: tq_model_load returns non-NULL handle for valid path\n");
    tq_model_free(model);
}

static void test_load_invalid_path(void) {
    tq_model_t model = tq_model_load("/nonexistent/path");
    assert(model == NULL);
    printf("  PASS: tq_model_load returns NULL for invalid path\n");
}

static void test_load_null_path(void) {
    tq_model_t model = tq_model_load(NULL);
    assert(model == NULL);
    printf("  PASS: tq_model_load returns NULL for NULL path\n");
}

static void test_free_null_safe(void) {
    tq_model_free(NULL); /* must not crash */
    printf("  PASS: tq_model_free(NULL) is safe\n");
}

static void test_version(void) {
    const char* v = tq_version();
    assert(v != NULL);
    assert(strncmp(v, "0.", 2) == 0);
    printf("  PASS: tq_version returns \"%s\"\n", v);
}

int main(void) {
    printf("test_c_api_model:\n");
    test_load_valid_model();
    test_load_invalid_path();
    test_load_null_path();
    test_free_null_safe();
    test_version();
    printf("All C API model tests passed.\n");
    return 0;
}
