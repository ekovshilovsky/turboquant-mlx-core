// Cache snapshot + delta tests. Validates serialization round-trips and the
// per-position slicing that backs distributed state transfer.

#include "turboquant/cache_snapshot.h"
#include "turboquant/kv_cache.h"

#include <cassert>
#include <cstdio>

using namespace turboquant;

static KVCacheConfig tiny_config(int num_layers, int num_heads, int head_dim) {
    KVCacheConfig cfg;
    cfg.num_layers = num_layers;
    cfg.num_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.k_bits = 3;
    cfg.v_bits = 3;
    cfg.max_context = 1024;
    cfg.decode_window = 128;
    return cfg;
}

static void append_random_token(TQKVCache& cache, int layer, int num_heads, int head_dim) {
    auto keys = mlx::core::random::normal({1, num_heads, head_dim});
    auto vals = mlx::core::random::normal({1, num_heads, head_dim});
    mlx::core::eval(keys, vals);
    cache.append(layer, keys, vals);
}

static void test_snapshot_roundtrip() {
    auto cfg = tiny_config(2, 4, 64);
    TQKVCache cache(cfg);

    for (int t = 0; t < 10; ++t) {
        for (int l = 0; l < cfg.num_layers; ++l) {
            append_random_token(cache, l, cfg.num_heads, cfg.head_dim);
        }
    }
    assert(cache.seq_length() == 10);

    auto snap = cache.snapshot();
    assert(snap.num_layers == 2u);
    assert(snap.num_positions == 10u);
    assert(snap.num_heads == 4u);
    assert(snap.head_dim == 64u);
    assert(snap.layer_data.size() == 2);

    TQKVCache restored(cfg);
    restored.restore(snap);
    assert(restored.seq_length() == 10);

    printf("  PASS: snapshot roundtrip\n");
}

static void test_delta_incremental() {
    auto cfg = tiny_config(1, 2, 32);
    TQKVCache source(cfg);

    for (int t = 0; t < 20; ++t) {
        append_random_token(source, 0, cfg.num_heads, cfg.head_dim);
    }
    assert(source.seq_length() == 20);

    auto delta = source.delta(10, 20);
    assert(delta.from_position == 10u);
    assert(delta.to_position == 20u);
    assert(delta.num_layers == 1u);
    assert(delta.layer_data.size() == 1);

    TQKVCache dest(cfg);
    for (int t = 0; t < 10; ++t) {
        append_random_token(dest, 0, cfg.num_heads, cfg.head_dim);
    }
    assert(dest.seq_length() == 10);

    dest.apply_delta(delta);
    assert(dest.seq_length() == 20);

    printf("  PASS: delta incremental\n");
}

static void test_snapshot_serialization_size() {
    auto cfg = tiny_config(1, 2, 32);
    TQKVCache cache(cfg);
    for (int t = 0; t < 5; ++t) {
        append_random_token(cache, 0, cfg.num_heads, cfg.head_dim);
    }

    auto snap = cache.snapshot();
    auto bytes = snapshot_serialize(snap);
    assert(bytes.size() > 0);
    assert(bytes.size() < 100000); // 5 tokens at 3-bit must remain small

    auto snap2 = snapshot_deserialize(bytes);
    assert(snap2.num_positions == 5u);
    assert(snap2.num_layers == 1u);
    assert(snap2.num_heads == 2u);
    assert(snap2.head_dim == 32u);
    assert(snap2.layer_data.size() == 1);
    assert(snap2.layer_data[0] == snap.layer_data[0]);

    printf("  PASS: snapshot serialization size\n");
}

int main() {
    printf("test_cache_snapshot:\n");
    test_snapshot_roundtrip();
    test_delta_incremental();
    test_snapshot_serialization_size();
    printf("All cache snapshot tests passed.\n");
    return 0;
}
