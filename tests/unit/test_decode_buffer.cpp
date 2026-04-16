#include "turboquant/decode_buffer.h"
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace turboquant;

static const int WINDOW  = 8;
static const int N_HEADS = 2;
static const int HEAD_DIM = 32;

// ---- helpers ----------------------------------------------------------------

/// Build a deterministic [num_tokens, N_HEADS, HEAD_DIM] float32 array.
static mlx::core::array make_kv(int num_tokens, float base_value) {
    std::vector<float> data(
        static_cast<size_t>(num_tokens * N_HEADS * HEAD_DIM));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = base_value + static_cast<float>(i) * 0.001f;
    }
    return mlx::core::array(
        data.data(), {num_tokens, N_HEADS, HEAD_DIM}, mlx::core::float32);
}

/// Build a single-token [1, N_HEADS, HEAD_DIM] key or value array.
static mlx::core::array make_single_token(int token_idx, float base_value) {
    return make_kv(1, base_value + static_cast<float>(token_idx) * 0.1f);
}

// ---- tests ------------------------------------------------------------------

/// Verify that rebuild sets current_length to the number of tokens provided.
static void test_rebuild_sets_current_length() {
    DecodeBuffer buf(WINDOW, N_HEADS, HEAD_DIM);
    assert(buf.current_length() == 0);

    const int n = 5;
    auto k = make_kv(n, 1.0f);
    auto v = make_kv(n, 2.0f);
    mlx::core::eval(k, v);

    buf.rebuild(k, v);
    assert(buf.current_length() == n);

    printf("  PASS: rebuild sets current_length correctly\n");
}

/// Verify that each append_token call increments current_length by exactly 1.
static void test_append_token_increments_length() {
    DecodeBuffer buf(WINDOW, N_HEADS, HEAD_DIM);

    for (int i = 0; i < 4; ++i) {
        int before = buf.current_length();
        auto k = make_single_token(i, 0.5f);
        auto v = make_single_token(i, 1.5f);
        mlx::core::eval(k, v);
        buf.append_token(k, v);
        assert(buf.current_length() == before + 1);
    }

    printf("  PASS: append_token increments current_length by 1 each call\n");
}

/// Verify that keys(start, end) returns an array shaped [end-start, N_HEADS, HEAD_DIM].
static void test_keys_values_shape() {
    DecodeBuffer buf(WINDOW, N_HEADS, HEAD_DIM);

    const int n = 6;
    auto k = make_kv(n, 1.0f);
    auto v = make_kv(n, 2.0f);
    mlx::core::eval(k, v);
    buf.rebuild(k, v);

    auto k_slice = buf.keys(1, 4);
    auto v_slice = buf.values(2, 6);
    mlx::core::eval(k_slice, v_slice);

    assert(k_slice.shape(0) == 3);
    assert(k_slice.shape(1) == N_HEADS);
    assert(k_slice.shape(2) == HEAD_DIM);

    assert(v_slice.shape(0) == 4);
    assert(v_slice.shape(1) == N_HEADS);
    assert(v_slice.shape(2) == HEAD_DIM);

    printf("  PASS: keys/values return correct [tokens, heads, head_dim] shape\n");
}

/// Verify that appending N tokens one-by-one and retrieving them via keys()
/// produces the same values as a single rebuild() with all N tokens.
static void test_incremental_append_matches_rebuild() {
    const int n = 5;

    // Build reference via rebuild
    DecodeBuffer ref_buf(WINDOW, N_HEADS, HEAD_DIM);
    auto all_k = make_kv(n, 3.0f);
    auto all_v = make_kv(n, 4.0f);
    mlx::core::eval(all_k, all_v);
    ref_buf.rebuild(all_k, all_v);

    auto ref_k = ref_buf.keys(0, n);
    auto ref_v = ref_buf.values(0, n);
    auto ref_k_f32 = mlx::core::astype(ref_k, mlx::core::float32);
    auto ref_v_f32 = mlx::core::astype(ref_v, mlx::core::float32);
    mlx::core::eval(ref_k_f32, ref_v_f32);

    // Build via per-token append starting from an empty buffer
    DecodeBuffer inc_buf(WINDOW, N_HEADS, HEAD_DIM);
    const size_t elems_per_token =
        static_cast<size_t>(N_HEADS * HEAD_DIM);

    auto all_k_f32 = mlx::core::astype(all_k, mlx::core::float32);
    auto all_v_f32 = mlx::core::astype(all_v, mlx::core::float32);
    mlx::core::eval(all_k_f32, all_v_f32);

    const float* ak = all_k_f32.data<float>();
    const float* av = all_v_f32.data<float>();

    for (int t = 0; t < n; ++t) {
        auto tk = mlx::core::array(ak + t * elems_per_token,
                                   {1, N_HEADS, HEAD_DIM},
                                   mlx::core::float32);
        auto tv = mlx::core::array(av + t * elems_per_token,
                                   {1, N_HEADS, HEAD_DIM},
                                   mlx::core::float32);
        mlx::core::eval(tk, tv);
        inc_buf.append_token(tk, tv);
    }

    auto inc_k = inc_buf.keys(0, n);
    auto inc_v = inc_buf.values(0, n);
    auto inc_k_f32 = mlx::core::astype(inc_k, mlx::core::float32);
    auto inc_v_f32 = mlx::core::astype(inc_v, mlx::core::float32);
    mlx::core::eval(inc_k_f32, inc_v_f32);

    // Compare element-by-element, allowing for float16 round-trip tolerance
    const size_t total = static_cast<size_t>(n) * elems_per_token;
    const float* rk = ref_k_f32.data<float>();
    const float* ik = inc_k_f32.data<float>();
    const float* rv = ref_v_f32.data<float>();
    const float* iv = inc_v_f32.data<float>();

    for (size_t i = 0; i < total; ++i) {
        // float32->float16->float32 round-trip error is at most ~1e-3 for
        // values in the range used here, so 1e-2 is a safe tolerance.
        assert(std::fabs(rk[i] - ik[i]) < 1e-2f &&
               "Key mismatch between incremental append and rebuild");
        assert(std::fabs(rv[i] - iv[i]) < 1e-2f &&
               "Value mismatch between incremental append and rebuild");
    }

    printf("  PASS: incremental append produces same values as rebuild\n");
}

/// Verify that appending tokens beyond window_size wraps the circular buffer
/// and that current_length continues to increment correctly.
static void test_circular_wrap_beyond_window() {
    DecodeBuffer buf(WINDOW, N_HEADS, HEAD_DIM);

    // Fill the buffer exactly
    for (int i = 0; i < WINDOW; ++i) {
        auto k = make_single_token(i, 0.0f);
        auto v = make_single_token(i, 1.0f);
        mlx::core::eval(k, v);
        buf.append_token(k, v);
    }
    assert(buf.current_length() == WINDOW);

    // Append three more tokens, wrapping around
    const int extra = 3;
    for (int i = 0; i < extra; ++i) {
        auto k = make_single_token(WINDOW + i, 5.0f);
        auto v = make_single_token(WINDOW + i, 6.0f);
        mlx::core::eval(k, v);
        buf.append_token(k, v);
    }
    assert(buf.current_length() == WINDOW + extra);

    // The most recent WINDOW tokens span positions [extra, WINDOW + extra).
    // We should be able to retrieve those without error.
    auto k_out = buf.keys(extra, WINDOW + extra);
    auto v_out = buf.values(extra, WINDOW + extra);
    mlx::core::eval(k_out, v_out);

    assert(k_out.shape(0) == WINDOW);
    assert(v_out.shape(0) == WINDOW);

    printf("  PASS: circular wrap increments length correctly and oldest"
           " slots are overwritten\n");
}

// ---- main -------------------------------------------------------------------

int main() {
    printf("test_decode_buffer:\n");
    test_rebuild_sets_current_length();
    test_append_token_increments_length();
    test_keys_values_shape();
    test_incremental_append_matches_rebuild();
    test_circular_wrap_beyond_window();
    printf("All decode buffer tests passed.\n");
    return 0;
}
