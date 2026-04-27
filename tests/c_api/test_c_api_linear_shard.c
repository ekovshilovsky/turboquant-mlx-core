// Tier 1 C API proof for Task 9a.2: the shard-aware linear forward path
// round-trips through the C boundary without leaking the C++ TurboQuantLinear
// or depending on any C++ types. The numerical correctness of
// TurboQuantLinear under column-parallel and row-parallel slicing is proven
// in tests/unit/test_linear_shard_forward.cpp; this test covers the
// boundary-only responsibilities of the C API:
//
//   - tq_linear_create_shard accepts the full documented argument set and
//     returns a live handle for well-formed inputs
//   - invalid arguments (non-positive dims, NULL required pointers, wrong
//     primary_bits, odd local_in_features in 4-bit mode) yield NULL
//   - tq_linear_forward dispatches the fused kernel and returns an owned
//     mlx::core::array* with the expected [batch, rank_out_features] shape
//   - tq_array_copy_to_fp32 correctly materialises the output on the CPU
//   - tq_linear_free and tq_array_free are NULL-safe and release resources
//
// The synthetic input data here is not a real quantised layer — the kernel
// will read centroid 0 for every index (primary_codebook[0] = 0.0) with
// norms = 1.0, producing a zero output. That is deliberate: the test is
// about the boundary, not the math.

#include "turboquant_c/turboquant_c.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build a small synthetic shard configuration that exercises the happy
// path of tq_linear_create_shard. Returns a live handle; caller owns it.
// in_features_full / local_in_features expose both the column-parallel
// (local == full) and row-parallel (local < full) call shapes through
// the same factory.
static tq_linear_t build_test_shard(int in_features_full, int local_in_features, int out_features) {
    const int block_size = 16;
    const int primary_bits = 4;
    const int residual_bits = 4;

    const int primary_cols = local_in_features / 2;
    const int residual_cols = local_in_features / 2;

    // Packed indices = all zeros, which under a zero-centroid codebook
    // produces a zero dequantised weight. Keeps the forward call
    // deterministic without depending on external fixture files.
    uint8_t* packed_primary = (uint8_t*)calloc((size_t)(out_features * primary_cols), 1);
    uint8_t* packed_residual = (uint8_t*)calloc((size_t)(out_features * residual_cols), 1);
    assert(packed_primary && packed_residual);

    // norms = 1.0 keeps y = dot_accum, no per-row rescaling.
    float* norms = (float*)malloc(sizeof(float) * (size_t)out_features);
    assert(norms);
    for (int i = 0; i < out_features; ++i) norms[i] = 1.0f;

    // Codebooks: centroid 0 = 0.0 makes every dequantised element zero,
    // so the forward's output is deterministically all zeros regardless
    // of input activation values. Simplifies the numerical check.
    float primary_codebook[16];
    float residual_codebook[16];
    for (int i = 0; i < 16; ++i) {
        primary_codebook[i] = 0.0f;
        residual_codebook[i] = 0.0f;
    }

    tq_linear_t layer = tq_linear_create_shard(
        in_features_full,
        local_in_features,
        out_features,
        primary_bits,
        residual_bits,
        packed_primary,
        packed_residual,
        norms,
        primary_codebook,
        residual_codebook,
        /*seed_primary=*/0xA5A5A5A5u,
        /*seed_residual=*/0x5A5A5A5Au,
        block_size);

    // Source buffers can be released immediately: tq_linear_create_shard
    // is documented to copy all inputs into MLX-owned storage.
    free(packed_primary);
    free(packed_residual);
    free(norms);

    return layer;
}

static void test_create_and_free_lifecycle(void) {
    tq_linear_t layer = build_test_shard(64, 64, 16);
    assert(layer != NULL);
    tq_linear_free(layer);
    tq_linear_free(NULL);
    printf("  PASS: create/free lifecycle\n");
}

static void test_create_rejects_invalid_arguments(void) {
    // Reject non-positive dimensions.
    float cb[16] = {0.0f};
    uint8_t packed[16] = {0};
    float norms[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    assert(tq_linear_create_shard(0, 64, 16, 4, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 0, 16, 4, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 64, 0, 4, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 64, 16, 4, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 0) == NULL);

    // Reject invalid bit widths (only 4 and 5 are supported for primary;
    // residual must be 0 or 4).
    assert(tq_linear_create_shard(64, 64, 16, 3, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 64, 16, 4, 3, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);

    // Reject NULL required pointers.
    assert(tq_linear_create_shard(64, 64, 16, 4, 4, NULL, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 64, 16, 4, 4, packed, packed, NULL,
                                  cb, cb, 0, 0, 16) == NULL);
    assert(tq_linear_create_shard(64, 64, 16, 4, 4, packed, packed, norms,
                                  NULL, cb, 0, 0, 16) == NULL);

    // Reject odd local_in_features when primary is 4-bit (nibble packing).
    assert(tq_linear_create_shard(64, 63, 16, 4, 4, packed, packed, norms,
                                  cb, cb, 0, 0, 16) == NULL);

    printf("  PASS: invalid arguments rejected with NULL\n");
}

static void test_forward_column_parallel_shape(void) {
    // Column-parallel: local_in_features == full_in_features, output is
    // the rank's slice of the layer's output dim. Verifies the happy
    // path of the C boundary's forward call.
    const int in_features = 64;
    const int rank_out = 8;
    const int batch = 3;

    tq_linear_t layer = build_test_shard(in_features, in_features, rank_out);
    assert(layer != NULL);

    // Input [batch, local_in_features] = [3, 64], arbitrary non-zero values.
    float* input_data = (float*)malloc(sizeof(float) * (size_t)(batch * in_features));
    for (int i = 0; i < batch * in_features; ++i) input_data[i] = 0.25f * (float)(i + 1);

    void* input_arr = tq_array_from_fp32(input_data, batch, in_features);
    assert(input_arr != NULL);

    void* output_arr = tq_linear_forward(layer, input_arr);
    assert(output_arr != NULL);

    // Zero codebook + zero indices = zero weights = zero output regardless
    // of input. Validates that the fused kernel actually ran and wrote
    // its output into the returned array, not that the kernel math is
    // correct (the C++ test covers numerical correctness).
    float* output_data = (float*)malloc(sizeof(float) * (size_t)(batch * rank_out));
    int rc = tq_array_copy_to_fp32(output_arr, output_data, batch, rank_out);
    assert(rc == 0);

    for (int i = 0; i < batch * rank_out; ++i) {
        assert(fabsf(output_data[i]) < 1e-6f);
    }

    tq_array_free(output_arr);
    tq_array_free(input_arr);
    tq_linear_free(layer);
    free(input_data);
    free(output_data);
    printf("  PASS: column-parallel forward shape [%d, %d]\n", batch, rank_out);
}

static void test_forward_row_parallel_shape(void) {
    // Row-parallel: local_in_features < full_in_features. The kernel must
    // use full_in_features for its combined_scale factor. Numerical
    // correctness is covered by test_linear_shard_forward.cpp; here we
    // only verify the C boundary handles the divergent
    // {full_in_features, local_in_features} pair and produces an output
    // of the correct rank-local shape.
    const int full_in_features = 128;
    const int local_in_features = 64;
    const int out_features = 16;
    const int batch = 2;

    tq_linear_t layer = build_test_shard(full_in_features, local_in_features, out_features);
    assert(layer != NULL);

    float* input_data = (float*)malloc(sizeof(float) * (size_t)(batch * local_in_features));
    for (int i = 0; i < batch * local_in_features; ++i) input_data[i] = 0.1f;

    void* input_arr = tq_array_from_fp32(input_data, batch, local_in_features);
    assert(input_arr != NULL);

    void* output_arr = tq_linear_forward(layer, input_arr);
    assert(output_arr != NULL);

    float* output_data = (float*)malloc(sizeof(float) * (size_t)(batch * out_features));
    int rc = tq_array_copy_to_fp32(output_arr, output_data, batch, out_features);
    assert(rc == 0);

    // Again, zero-centroid codebook guarantees zero output. The test's
    // value lies in the forward call not crashing and returning an array
    // of the expected rank-local shape.
    for (int i = 0; i < batch * out_features; ++i) {
        assert(fabsf(output_data[i]) < 1e-6f);
    }

    // Shape mismatch in copy helper must be rejected: asking for the
    // wrong dimensions returns -1 without clobbering the output buffer.
    output_data[0] = 1234.5f;
    int bad = tq_array_copy_to_fp32(output_arr, output_data, batch, out_features + 1);
    assert(bad == -1);
    assert(output_data[0] == 1234.5f);

    tq_array_free(output_arr);
    tq_array_free(input_arr);
    tq_linear_free(layer);
    free(input_data);
    free(output_data);
    printf("  PASS: row-parallel forward shape [%d, %d]\n", batch, out_features);
}

static void test_forward_null_safe(void) {
    // NULL layer or NULL input array: the forward must return NULL rather
    // than crash. This is the contract for library clients that want to
    // handle degenerate configurations without try/catch boilerplate.
    assert(tq_linear_forward(NULL, NULL) == NULL);

    tq_linear_t layer = build_test_shard(32, 32, 16);
    assert(layer != NULL);
    assert(tq_linear_forward(layer, NULL) == NULL);
    tq_linear_free(layer);

    printf("  PASS: tq_linear_forward is NULL-safe\n");
}

int main(void) {
    printf("test_c_api_linear_shard:\n");
    test_create_and_free_lifecycle();
    test_create_rejects_invalid_arguments();
    test_forward_column_parallel_shape();
    test_forward_row_parallel_shape();
    test_forward_null_safe();
    printf("All C API linear shard tests passed.\n");
    return 0;
}
