/// Test MLX GPU stream behavior for concurrent dispatch.
///
/// FINDING: MLX new_stream() creates logically separate streams but they
/// share Metal command queue internals. Concurrent eval() from 2+ threads
/// causes a segfault even with separate streams. Single-thread dispatch
/// to multiple streams (main thread only) works correctly.
///
/// This means GPU acceleration in the converter must use a single-threaded
/// GPU streaming pipeline, not per-thread GPU dispatch.

#include <mlx/mlx.h>
#include <cassert>
#include <cstdio>

namespace mx = mlx::core;

static void test_create_multiple_streams() {
    auto s1 = mx::new_stream(mx::Device::gpu);
    auto s2 = mx::new_stream(mx::Device::gpu);
    auto s3 = mx::new_stream(mx::Device::gpu);

    assert(s1 != s2);
    assert(s2 != s3);
    assert(s1 != s3);
    printf("  PASS: 3 independent GPU streams (indices %d, %d, %d)\n",
           s1.index, s2.index, s3.index);
}

static void test_single_thread_multi_stream() {
    auto s1 = mx::new_stream(mx::Device::gpu);
    auto s2 = mx::new_stream(mx::Device::gpu);

    auto a = mx::random::normal({500, 500});
    auto b = mx::random::normal({500, 500});

    // Dispatch to different streams from the SAME thread — this works
    auto c = mx::matmul(a, b, s1);
    auto d = mx::matmul(b, a, s2);
    mx::eval(c, d);

    assert(c.shape(0) == 500 && c.shape(1) == 500);
    assert(d.shape(0) == 500 && d.shape(1) == 500);
    printf("  PASS: single-thread dispatch to 2 GPU streams\n");
}

static void test_sequential_stream_reuse() {
    // Verify streams can be reused across multiple dispatches
    auto s = mx::new_stream(mx::Device::gpu);

    for (int i = 0; i < 5; i++) {
        auto x = mx::random::normal({200, 200});
        auto r = mx::matmul(x, x, s);
        mx::eval(r);
        assert(r.shape(0) == 200);
    }
    printf("  PASS: stream reuse across 5 sequential dispatches\n");
}

/// NOTE: concurrent eval() from multiple threads with separate streams
/// causes a segfault in MLX 0.31.1. This is documented here but NOT
/// tested because it would crash the test runner.
///
/// The segfault occurs in MLX's Metal backend when two threads
/// simultaneously call eval() — even with different streams, they
/// share the underlying command buffer encoder which is not thread-safe.
///
/// Verified: 1 thread with GPU stream works. 2 concurrent threads crash.

int main() {
    printf("test_gpu_streams:\n");
    test_create_multiple_streams();
    test_single_thread_multi_stream();
    test_sequential_stream_reuse();
    printf("All GPU stream tests passed.\n");
    printf("\nNOTE: per-thread concurrent GPU eval() crashes MLX 0.31.1.\n");
    printf("Converter uses single-threaded GPU + multi-threaded CPU as workaround.\n");
    return 0;
}
