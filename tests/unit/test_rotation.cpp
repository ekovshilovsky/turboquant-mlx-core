#include "turboquant/quantizer.h"
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace turboquant;

/// Compute the L2 norm of a 1-D float array.
static float l2_norm(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += data[i] * data[i];
    return std::sqrt(sum);
}

/// Verify that inverse_WHT(WHT(x)) recovers the original input within
/// a tolerance of 1e-4 per element.
static void test_wht_is_self_inverse() {
    const int rows = 1, cols = 512;
    auto x = mlx::core::random::normal({rows, cols});
    mlx::core::eval(x);

    uint32_t seed = 42;
    auto rotated   = apply_wht_rotation(x, seed, cols);
    auto recovered = apply_inverse_wht_rotation(rotated, seed, cols);
    mlx::core::eval(rotated, recovered);

    const float* src = x.data<float>();
    const float* rec = recovered.data<float>();
    for (int i = 0; i < rows * cols; ++i) {
        float err = std::abs(src[i] - rec[i]);
        assert(err < 1e-4f && "self-inverse round-trip error exceeded 1e-4");
    }
    printf("  PASS: WHT is self-inverse\n");
}

/// Verify that the WHT is norm-preserving: ||WHT(x)|| / ||x|| == 1.0 +/- 1e-3.
static void test_wht_preserves_norm() {
    const int rows = 1, cols = 512;
    auto x = mlx::core::random::normal({rows, cols});
    auto rotated = apply_wht_rotation(x, 42, cols);
    mlx::core::eval(x, rotated);

    float norm_x = l2_norm(x.data<float>(), cols);
    float norm_r = l2_norm(rotated.data<float>(), cols);
    float ratio  = norm_r / norm_x;
    assert(std::abs(ratio - 1.0f) < 1e-3f && "WHT is not norm-preserving");
    printf("  PASS: WHT preserves L2 norm\n");
}

/// Verify that calling apply_wht_rotation twice with the same seed yields
/// bit-identical results.
static void test_seed_determinism() {
    const int rows = 1, cols = 512;
    auto x  = mlx::core::random::normal({rows, cols});
    auto r1 = apply_wht_rotation(x, 42, cols);
    auto r2 = apply_wht_rotation(x, 42, cols);
    mlx::core::eval(r1, r2);

    const float* d1 = r1.data<float>();
    const float* d2 = r2.data<float>();
    for (int i = 0; i < rows * cols; ++i) {
        assert(d1[i] == d2[i] && "same seed produced non-identical outputs");
    }
    printf("  PASS: same seed produces identical rotation\n");
}

/// Verify that two different seeds produce different rotation outputs.
static void test_different_seeds_differ() {
    const int rows = 1, cols = 512;
    auto x  = mlx::core::random::normal({rows, cols});
    auto r1 = apply_wht_rotation(x, 42, cols);
    auto r2 = apply_wht_rotation(x, 99, cols);
    mlx::core::eval(r1, r2);

    const float* d1 = r1.data<float>();
    const float* d2 = r2.data<float>();
    bool differ = false;
    for (int i = 0; i < rows * cols; ++i) {
        if (d1[i] != d2[i]) { differ = true; break; }
    }
    assert(differ && "different seeds produced identical outputs");
    printf("  PASS: different seeds produce different rotations\n");
}

/// Verify that a 384-dim input (3 x 128-element blocks) round-trips correctly.
static void test_block_chunking_384_dim() {
    const int rows = 1, cols = 384, block_size = 128;
    auto x = mlx::core::random::normal({rows, cols});
    mlx::core::eval(x);

    auto rotated   = apply_wht_rotation(x, 42, block_size);
    auto recovered = apply_inverse_wht_rotation(rotated, 42, block_size);
    mlx::core::eval(rotated, recovered);

    const float* src = x.data<float>();
    const float* rec = recovered.data<float>();
    for (int i = 0; i < rows * cols; ++i) {
        float err = std::abs(src[i] - rec[i]);
        assert(err < 1e-4f && "block-chunked round-trip error exceeded 1e-4");
    }
    printf("  PASS: block chunking handles 384-dim (3x128)\n");
}

/// Verify round-trip rotation on non-power-of-2 dimensions from real models.
/// 896 = 7 x 128 (Qwen2.5-0.5B hidden_size) and 4864 = 19 x 256 are dimensions
/// that previously triggered a remainder bug where trailing columns were left
/// unrotated because the column count is not divisible by 512.
static void test_non_power_of_2_round_trip() {
    // 896-dim with block_size=128 (Qwen2.5-0.5B hidden_size: 896 = 7 x 128)
    {
        const int rows = 4, cols = 896, block_size = 128;
        auto x = mlx::core::random::normal({rows, cols});
        mlx::core::eval(x);

        auto rotated   = apply_wht_rotation(x, 42, block_size);
        auto recovered = apply_inverse_wht_rotation(rotated, 42, block_size);
        mlx::core::eval(rotated, recovered);

        const float* src = x.data<float>();
        const float* rec = recovered.data<float>();
        for (int i = 0; i < rows * cols; ++i) {
            float err = std::abs(src[i] - rec[i]);
            assert(err < 1e-4f && "896-dim round-trip error exceeded 1e-4");
        }
        printf("    896-dim (7x128): OK\n");
    }

    // 4864-dim with block_size=256 (4864 = 19 x 256)
    {
        const int rows = 4, cols = 4864, block_size = 256;
        auto x = mlx::core::random::normal({rows, cols});
        mlx::core::eval(x);

        auto rotated   = apply_wht_rotation(x, 42, block_size);
        auto recovered = apply_inverse_wht_rotation(rotated, 42, block_size);
        mlx::core::eval(rotated, recovered);

        const float* src = x.data<float>();
        const float* rec = recovered.data<float>();
        for (int i = 0; i < rows * cols; ++i) {
            float err = std::abs(src[i] - rec[i]);
            assert(err < 1e-4f && "4864-dim round-trip error exceeded 1e-4");
        }
        printf("    4864-dim (19x256): OK\n");
    }

    printf("  PASS: non-power-of-2 dimensions round-trip correctly\n");
}

int main() {
    printf("test_rotation:\n");
    test_wht_is_self_inverse();
    test_wht_preserves_norm();
    test_seed_determinism();
    test_different_seeds_differ();
    test_block_chunking_384_dim();
    test_non_power_of_2_round_trip();
    printf("All rotation tests passed.\n");
    return 0;
}
