// Tier 1 C++ proof for Task 9a.2: TurboQuantLinear handles rank-local
// shards for both column-parallel and row-parallel tensor-parallel splits
// of a TurboQuant-quantised weight matrix.
//
// 9a.1 established that TurboQuant's rotation is block-diagonal along the
// input dimension, so slicing the packed arrays on group-aligned
// boundaries preserves the kernel's correctness. The row-parallel slice
// additionally requires the fused kernel to know the pre-shard
// in_features, otherwise combined_scale is sqrt(N)x too large. This
// test exercises both paths end-to-end through the existing C++ class.
//
// Assertions:
//   (1) Column-parallel: the rank's forward output equals the
//       corresponding output-row slice of the whole-weight forward.
//       Verifies that subsetting packed arrays along the output dim
//       leaves the fused kernel's scaling untouched.
//   (2) Row-parallel: summing the per-rank partial forwards (each
//       invoked on its input-dim slice, with full_in_features passed
//       through the shard-aware constructor) equals the whole-weight
//       forward. Verifies that the Metal kernel's new full_in_features
//       parameter restores correctness for row-parallel shards.
//
// Tolerance: 1e-3. The WHT factorization is exact in float arithmetic
// (algebraic rel_frob ~1e-7 per 9a.1 finding); the observed error is
// dominated by the fp16 accumulation path inside the fused kernel.

#include "turboquant/codebook.h"
#include "turboquant/linear.h"
#include "turboquant/quantizer.h"

#include <mlx/mlx.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace turboquant;

namespace {

// Deterministic uniform-in-[-1,1] matrix generator. Using a hand-rolled
// LCG keeps the test reproducible across MLX versions without requiring
// global seeding of MLX's RNG.
mlx::core::array make_matrix(int rows, int cols, uint32_t seed) {
    std::vector<float> data(static_cast<size_t>(rows) * cols);
    uint32_t state = seed;
    for (size_t i = 0; i < data.size(); ++i) {
        state = state * 1103515245u + 12345u;
        data[i] = (static_cast<float>(state >> 16) / 32768.0f) - 1.0f;
    }
    return mlx::core::array(data.data(), {rows, cols}, mlx::core::float32);
}

float relative_frobenius_error(const mlx::core::array& actual,
                               const mlx::core::array& expected) {
    auto diff = mlx::core::subtract(actual, expected);
    auto num = mlx::core::sqrt(mlx::core::sum(mlx::core::multiply(diff, diff)));
    auto den = mlx::core::sqrt(mlx::core::sum(
        mlx::core::multiply(expected, expected)));
    mlx::core::eval(num);
    mlx::core::eval(den);
    const float d = den.item<float>();
    if (d < 1e-12f) return num.item<float>();
    return num.item<float>() / d;
}

float max_abs_diff(const mlx::core::array& a, const mlx::core::array& b) {
    auto diff = mlx::core::subtract(a, b);
    auto abs_diff = mlx::core::abs(diff);
    auto max_val = mlx::core::max(abs_diff);
    mlx::core::eval(max_val);
    return max_val.item<float>();
}

// Slice a QuantizedWeight along the OUTPUT dimension [row_lo, row_hi).
// Column-parallel shards hold a contiguous band of output rows. Seeds
// and block_size are layer-global; the norms and packed arrays carry
// per-row values that we subset directly. The fused kernel reads the
// packed buffers via flat raw-pointer indexing, so each shard's view
// must be row-contiguous — mlx::core::contiguous copies the sliced
// view into a fresh dense allocation.
QuantizedWeight slice_output_rows(const QuantizedWeight& qw,
                                  int row_lo,
                                  int row_hi) {
    const int packed_cols = static_cast<int>(qw.packed_primary.shape(1));
    const int residual_packed_cols =
        static_cast<int>(qw.packed_residual.shape(1));

    auto slice_rows = [&](const mlx::core::array& arr, int last_dim) {
        auto sliced = mlx::core::contiguous(
            mlx::core::slice(arr, {row_lo, 0}, {row_hi, last_dim}));
        mlx::core::eval(sliced);
        return sliced;
    };

    auto slice_norms = [&](const mlx::core::array& arr) {
        auto sliced = mlx::core::contiguous(
            mlx::core::slice(arr, {row_lo}, {row_hi}));
        mlx::core::eval(sliced);
        return sliced;
    };

    return QuantizedWeight{
        slice_rows(qw.packed_primary, packed_cols),
        slice_rows(qw.packed_residual, residual_packed_cols),
        slice_norms(qw.norms),
        qw.seeds,
    };
}

// Slice a QuantizedWeight along the INPUT dimension [col_lo, col_hi).
// Row-parallel shards hold a contiguous band of input-dim elements;
// boundaries must be even to respect the 4-bit nibble packing and
// must be multiples of block_size (9a.1 group-alignment constraint).
// Seeds and norms are broadcast verbatim — the per-row norm
// correction was calibrated against the full reconstruction and is
// identical across row-parallel shards by construction. The fused
// kernel indexes packed arrays by flat offset, so the shard's sliced
// view must be materialised as a fresh contiguous buffer.
QuantizedWeight slice_input_cols(const QuantizedWeight& qw,
                                 int col_lo,
                                 int col_hi) {
    assert(col_lo % 2 == 0 && col_hi % 2 == 0 &&
           "column boundaries must be even for 4-bit nibble packing");

    const int rows = static_cast<int>(qw.packed_primary.shape(0));
    const int packed_lo = col_lo / 2;
    const int packed_hi = col_hi / 2;

    auto slice_cols = [&](const mlx::core::array& arr) {
        auto sliced = mlx::core::contiguous(
            mlx::core::slice(arr, {0, packed_lo}, {rows, packed_hi}));
        mlx::core::eval(sliced);
        return sliced;
    };

    return QuantizedWeight{
        slice_cols(qw.packed_primary),
        slice_cols(qw.packed_residual),
        qw.norms,
        qw.seeds,
    };
}

} // namespace

static void test_column_parallel_shard_matches_full_slice() {
    constexpr int batch = 2;
    constexpr int in_features = 128;
    constexpr int out_features = 16;
    constexpr uint32_t block_size = 64;
    constexpr int n_shards = 2;
    constexpr int shard_out = out_features / n_shards;

    auto weight = make_matrix(out_features, in_features, /*seed=*/0x9a2c01);
    auto input  = make_matrix(batch, in_features, /*seed=*/0x9a2c02);
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig cfg{};
    cfg.primary_bits = 4; cfg.residual_bits = 4;
    cfg.block_size = block_size; cfg.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, cfg);

    TurboQuantLinear whole(in_features, out_features, qw,
                           primary_cb, residual_cb, block_size);
    auto y_full = mlx::core::astype(whole.forward(input), mlx::core::float32);
    mlx::core::eval(y_full);

    for (int s = 0; s < n_shards; ++s) {
        const int lo = s * shard_out;
        const int hi = lo + shard_out;
        auto qw_shard = slice_output_rows(qw, lo, hi);

        // Column-parallel shards keep the full input dim, so full_in_features
        // defaults to in_features — scaling is unchanged vs the whole-weight
        // forward and the output must match the corresponding row slice
        // exactly up to fp16 accumulation error.
        TurboQuantLinear shard_linear(in_features, shard_out, qw_shard,
                                      primary_cb, residual_cb, block_size);
        auto y_shard_f16 = shard_linear.forward(input);
        auto y_shard = mlx::core::astype(y_shard_f16, mlx::core::float32);
        mlx::core::eval(y_shard);

        auto y_full_slice = mlx::core::slice(y_full, {0, lo}, {batch, hi});
        mlx::core::eval(y_full_slice);

        const float rel = relative_frobenius_error(y_shard, y_full_slice);
        const float ma = max_abs_diff(y_shard, y_full_slice);
        printf("  [column-parallel] shard=%d rows=[%d,%d) rel_frob=%.3e "
               "max_abs=%.3e\n", s, lo, hi, rel, ma);

        assert(rel < 1e-3f);
    }
    printf("  PASS: column-parallel shards match whole-weight row slices\n");
}

static void test_row_parallel_shards_sum_to_full_forward() {
    constexpr int batch = 2;
    constexpr int in_features = 128;
    constexpr int out_features = 16;
    constexpr uint32_t block_size = 64;
    constexpr int n_shards = 2;
    constexpr int shard_in = in_features / n_shards;

    static_assert(in_features % (n_shards * static_cast<int>(block_size)) == 0,
                  "row-parallel shard boundaries must be multiples of "
                  "block_size per 9a.1 group-alignment finding");

    auto weight = make_matrix(out_features, in_features, /*seed=*/0x9a2c11);
    auto input  = make_matrix(batch, in_features, /*seed=*/0x9a2c12);
    auto primary_cb  = generate_codebook(4);
    auto residual_cb = generate_codebook(4);

    QuantizerConfig cfg{};
    cfg.primary_bits = 4; cfg.residual_bits = 4;
    cfg.block_size = block_size; cfg.norm_correction = true;

    auto qw = quantize_weight(weight, primary_cb, residual_cb, cfg);

    TurboQuantLinear whole(in_features, out_features, qw,
                           primary_cb, residual_cb, block_size);
    auto y_full = mlx::core::astype(whole.forward(input), mlx::core::float32);
    mlx::core::eval(y_full);

    mlx::core::array y_sum = mlx::core::zeros_like(y_full);
    mlx::core::eval(y_sum);

    for (int s = 0; s < n_shards; ++s) {
        const int lo = s * shard_in;
        const int hi = lo + shard_in;
        auto qw_shard = slice_input_cols(qw, lo, hi);

        // Row-parallel shard: local in_features = shard_in but the
        // offline-quantised norms were calibrated against the full
        // in_features. Pass full_in_features so the kernel's
        // combined_scale matches the whole-weight invocation.
        TurboQuantLinear shard_linear(shard_in, out_features, qw_shard,
                                      primary_cb, residual_cb, block_size,
                                      /*full_in_features=*/in_features);

        auto x_shard = mlx::core::contiguous(
            mlx::core::slice(input, {0, lo}, {batch, hi}));
        mlx::core::eval(x_shard);

        auto y_shard_f16 = shard_linear.forward(x_shard);
        auto y_shard = mlx::core::astype(y_shard_f16, mlx::core::float32);
        mlx::core::eval(y_shard);

        y_sum = mlx::core::add(y_sum, y_shard);
        mlx::core::eval(y_sum);
    }

    const float rel = relative_frobenius_error(y_sum, y_full);
    const float ma = max_abs_diff(y_sum, y_full);
    printf("  [row-parallel] N=%d rel_frob=%.3e max_abs=%.3e\n",
           n_shards, rel, ma);

    assert(rel < 1e-3f);
    printf("  PASS: row-parallel shard partials sum to whole-weight forward\n");
}

int main() {
    printf("test_linear_shard_forward:\n");
    test_column_parallel_shard_matches_full_slice();
    test_row_parallel_shards_sum_to_full_forward();
    printf("All TurboQuantLinear shard forward tests passed.\n");
    return 0;
}
