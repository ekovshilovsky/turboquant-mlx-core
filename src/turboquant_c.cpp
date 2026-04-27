#include "turboquant_c/turboquant_c.h"
#include "turboquant/turboquant.h"
#include "turboquant/kv_cache.h"
#include "turboquant/distributed.h"
#include "turboquant/linear.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace turboquant {

const char* version() {
    return "0.1.0";
}

} // namespace turboquant

/// Opaque handle for a loaded TurboQuant model.
/// Full model loading (multi-layer TurboQuantLinear graph construction) is
/// deferred until SwiftLM integration provides the model architecture metadata.
/// For now this handle validates the model path and tracks loaded state.
struct tq_model_s {
    std::string model_path;
    bool loaded;
};

/// Opaque handle wrapping TQKVCache for the C boundary.
struct tq_kv_cache_s {
    turboquant::TQKVCache* cache;
};

/// Opaque handle wrapping TQDistributedCoordinator for the C boundary.
struct tq_coordinator_s {
    turboquant::TQDistributedCoordinator* coordinator;
};

/// Opaque handle for builder-pattern memory-aware cluster planning.
/// Accumulates discovered peers via tq_cluster_add_node and retains the
/// computed ShardPlan so per-rank layer ranges can be queried through the
/// C boundary without exposing std::vector or C++ types.
struct tq_cluster_s {
    int num_layers;
    int num_heads;
    int head_dim;
    std::vector<turboquant::NodeMemoryInfo> nodes;
    turboquant::ShardPlan plan;
    bool planned = false;
};

/// Opaque handle wrapping a rank-local TurboQuantLinear. Owns its C++
/// state so raw pointers passed through tq_linear_create_shard can be
/// released by the caller immediately after construction: the underlying
/// mlx::core::array wrappers copy input data during construction (via the
/// array(T* data, shape, dtype) ctor) so the layer is self-contained
/// against any input-buffer lifetime concerns on the Swift side.
///
/// TurboQuantLinear stores the QuantizedWeight and both Codebooks by
/// value, so wrapping it in unique_ptr keeps the handle move-only without
/// exposing any C++ types across the C boundary.
struct tq_linear_s {
    std::unique_ptr<turboquant::TurboQuantLinear> layer;
};

extern "C" {

// ---------------------------------------------------------------------------
// Model API
// ---------------------------------------------------------------------------

tq_model_t tq_model_load(const char* model_path) {
    try {
        if (!model_path) return nullptr;
        if (!std::filesystem::exists(model_path)) return nullptr;
        if (!std::filesystem::is_directory(model_path)) return nullptr;

        auto* model = new tq_model_s();
        model->model_path = model_path;
        model->loaded = true;
        return model;
    } catch (...) {
        return nullptr;
    }
}

void* tq_model_forward(tq_model_t model, const void* input_array) {
    // Stubbed: full forward pass requires SwiftLM model architecture integration
    // to construct the layer graph from TurboQuantLinear modules.
    (void)model;
    (void)input_array;
    return nullptr;
}

void tq_model_free(tq_model_t model) {
    if (!model) return;
    delete model;
}

// ---------------------------------------------------------------------------
// KV Cache API
// ---------------------------------------------------------------------------

tq_kv_cache_t tq_kv_cache_create(
    int num_layers, int num_heads, int head_dim,
    int kv_bits, int max_context, int decode_window) {
    try {
        turboquant::KVCacheConfig config;
        config.num_layers    = num_layers;
        config.num_heads     = num_heads;
        config.head_dim      = head_dim;
        config.k_bits        = static_cast<uint8_t>(kv_bits);
        config.v_bits        = static_cast<uint8_t>(kv_bits);
        config.max_context   = max_context;
        config.decode_window = decode_window;

        auto* cache = new turboquant::TQKVCache(config);
        auto* handle = new tq_kv_cache_s();
        handle->cache = cache;
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void tq_kv_cache_append(
    tq_kv_cache_t cache, int layer,
    const void* keys_array, const void* values_array) {
    if (!cache || !cache->cache || !keys_array || !values_array) return;
    try {
        const auto* keys = static_cast<const mlx::core::array*>(keys_array);
        const auto* values = static_cast<const mlx::core::array*>(values_array);
        cache->cache->append(layer, *keys, *values);
    } catch (...) {
        // Swallow C++ exceptions at the C boundary
    }
}

void* tq_kv_cache_get_keys_fp16(tq_kv_cache_t cache, int layer, int start, int end) {
    if (!cache || !cache->cache) return nullptr;
    try {
        auto result = cache->cache->get_keys_fp16(layer, start, end);
        return new mlx::core::array(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

void* tq_kv_cache_get_values_fp16(tq_kv_cache_t cache, int layer, int start, int end) {
    if (!cache || !cache->cache) return nullptr;
    try {
        auto result = cache->cache->get_values_fp16(layer, start, end);
        return new mlx::core::array(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

void* tq_kv_cache_attention(
    tq_kv_cache_t cache, int layer,
    const void* queries_array,
    int compressed_start, int compressed_end) {
    if (!cache || !cache->cache || !queries_array) return nullptr;
    try {
        const auto* queries = static_cast<const mlx::core::array*>(queries_array);
        auto result = cache->cache->compressed_attention(
            layer, *queries, compressed_start, compressed_end);
        return new mlx::core::array(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

int tq_kv_cache_seq_length(tq_kv_cache_t cache) {
    if (!cache || !cache->cache) return 0;
    return cache->cache->seq_length();
}

void tq_kv_cache_free(tq_kv_cache_t cache) {
    if (!cache) return;
    delete cache->cache;
    delete cache;
}

// ---------------------------------------------------------------------------
// Distributed API
// ---------------------------------------------------------------------------

tq_coordinator_t tq_distributed_init(const char* hostfile, const char* backend) {
    try {
        if (!hostfile) return nullptr;

        auto* coordinator = new turboquant::TQDistributedCoordinator();
        std::string be = backend ? backend : "auto";
        if (!coordinator->init(hostfile, be)) {
            delete coordinator;
            return nullptr;
        }

        auto* handle = new tq_coordinator_s();
        handle->coordinator = coordinator;
        return handle;
    } catch (...) {
        return nullptr;
    }
}

tq_coordinator_t tq_distributed_init_local(void) {
    try {
        auto* coordinator = new turboquant::TQDistributedCoordinator();
        if (!coordinator->init_local()) {
            delete coordinator;
            return nullptr;
        }

        auto* handle = new tq_coordinator_s();
        handle->coordinator = coordinator;
        return handle;
    } catch (...) {
        return nullptr;
    }
}

int tq_distributed_rank(tq_coordinator_t coord) {
    if (!coord || !coord->coordinator) return 0;
    return coord->coordinator->rank();
}

int tq_distributed_world_size(tq_coordinator_t coord) {
    if (!coord || !coord->coordinator) return 1;
    return coord->coordinator->world_size();
}

void tq_distributed_free(tq_coordinator_t coord) {
    if (!coord) return;
    delete coord->coordinator;
    delete coord;
}

// ---------------------------------------------------------------------------
// Cluster API
// ---------------------------------------------------------------------------
// Builder-pattern wrapper over TQDistributedCoordinator::plan_memory_aware.
// Swift drives peer discovery through Bonjour, accumulates each peer's
// reported usable memory via tq_cluster_add_node, then resolves a single
// pipeline-parallel assignment with tq_cluster_plan. This keeps the pure
// planning math reachable from the C boundary without requiring a hostfile
// or bringing up a live collective backend.

tq_cluster_t tq_cluster_create(int num_layers, int num_heads, int head_dim) {
    try {
        if (num_layers <= 0 || num_heads <= 0 || head_dim <= 0) return nullptr;
        auto* handle = new tq_cluster_s();
        handle->num_layers = num_layers;
        handle->num_heads  = num_heads;
        handle->head_dim   = head_dim;
        return handle;
    } catch (...) {
        return nullptr;
    }
}

int tq_cluster_add_node(tq_cluster_t cluster, const char* hostname, double memory_gb) {
    if (!cluster || !hostname) return -1;
    if (memory_gb <= 0.0) return -1;
    // Reject node registration after planning has been performed so the
    // plan and the registered node list stay consistent for query calls.
    if (cluster->planned) return -1;
    try {
        turboquant::NodeMemoryInfo info;
        info.hostname = hostname;
        info.usable_memory_gb = memory_gb;
        cluster->nodes.push_back(std::move(info));
        return 0;
    } catch (...) {
        return -1;
    }
}

int tq_cluster_plan(tq_cluster_t cluster) {
    if (!cluster) return -1;
    if (cluster->planned) return -1;
    if (cluster->nodes.empty()) return -1;
    try {
        // init_local suffices here: planning is a pure function over the
        // supplied NodeMemoryInfo list and does not require an active
        // collective backend. Distributed runtime setup happens later,
        // once the plan has been distributed to each peer.
        turboquant::TQDistributedCoordinator coord;
        if (!coord.init_local()) return -1;
        cluster->plan = coord.plan_memory_aware(
            cluster->num_layers, cluster->num_heads, cluster->head_dim,
            cluster->nodes);
        cluster->planned = true;
        return 0;
    } catch (...) {
        return -1;
    }
}

int tq_cluster_node_count(tq_cluster_t cluster) {
    if (!cluster) return 0;
    return static_cast<int>(cluster->nodes.size());
}

int tq_cluster_get_layer_start(tq_cluster_t cluster, int rank) {
    if (!cluster || !cluster->planned) return -1;
    if (rank < 0 || rank >= static_cast<int>(cluster->plan.assignments.size())) return -1;
    return cluster->plan.assignments[rank].layer_start;
}

int tq_cluster_get_layer_end(tq_cluster_t cluster, int rank) {
    if (!cluster || !cluster->planned) return -1;
    if (rank < 0 || rank >= static_cast<int>(cluster->plan.assignments.size())) return -1;
    return cluster->plan.assignments[rank].layer_end;
}

void tq_cluster_free(tq_cluster_t cluster) {
    if (!cluster) return;
    delete cluster;
}

// ---------------------------------------------------------------------------
// Shard-aware Linear API
// ---------------------------------------------------------------------------
// Each handle wraps a rank-local TurboQuantLinear built from this rank's
// slice of packed_primary / packed_residual / norms. The mlx::core::array
// instances constructed here copy the inbound data, so callers can free
// their source buffers immediately after tq_linear_create_shard returns.
//
// For row-parallel shards, local_in_features is this rank's slice of the
// input dim and full_in_features is the layer's original in_features.
// TurboQuantLinear threads full_in_features through to the fused kernel's
// combined_scale so the per-row norm correction (calibrated against the
// whole layer) aligns with the rank-local dequantised weight values.
// For column-parallel shards and whole-weight calls, local and full match
// and the kernel behaves exactly as the pre-shard path.

tq_linear_t tq_linear_create_shard(
    int full_in_features,
    int local_in_features,
    int rank_out_features,
    int primary_bits,
    int residual_bits,
    const void* packed_primary,
    const void* packed_residual,
    const void* norms,
    const void* primary_codebook,
    const void* residual_codebook,
    unsigned int seed_primary,
    unsigned int seed_residual,
    int block_size) {
    try {
        if (full_in_features <= 0 || local_in_features <= 0
            || rank_out_features <= 0 || block_size <= 0) {
            return nullptr;
        }
        if (primary_bits != 4 && primary_bits != 5) return nullptr;
        if (residual_bits != 0 && residual_bits != 4) return nullptr;
        if (!packed_primary || !norms || !primary_codebook) return nullptr;

        // 4-bit primary packs two nibble indices per byte; 5-bit primary
        // stores one index per byte. The kernel detects the packing mode
        // by comparing primary packed_cols to the residual's (always 4-bit
        // packed) columns, so local_in_features must be even whenever the
        // residual buffer is non-null.
        const int primary_packed_cols =
            (primary_bits == 4) ? local_in_features / 2 : local_in_features;
        const int residual_packed_cols = local_in_features / 2;
        if (primary_bits == 4 && (local_in_features & 1)) return nullptr;
        if (residual_bits == 4 && (local_in_features & 1)) return nullptr;

        const int primary_codebook_size = 1 << primary_bits;
        const int residual_codebook_size = (residual_bits > 0) ? (1 << residual_bits) : 0;

        // Build the primary Codebook from the caller-provided centroids.
        turboquant::Codebook primary_cb;
        primary_cb.bits = static_cast<uint8_t>(primary_bits);
        primary_cb.centroids.assign(
            static_cast<const float*>(primary_codebook),
            static_cast<const float*>(primary_codebook) + primary_codebook_size);
        primary_cb.boundaries.clear();
        primary_cb.origin = turboquant::CodebookOrigin::Empirical;

        // Residual codebook may be absent: provide a minimal stub so the
        // fused kernel's always-read path does not fault. The kernel
        // inspects packed_residual to detect whether residual is active.
        turboquant::Codebook residual_cb;
        residual_cb.bits = static_cast<uint8_t>(residual_bits > 0 ? residual_bits : 4);
        if (residual_codebook && residual_codebook_size > 0) {
            residual_cb.centroids.assign(
                static_cast<const float*>(residual_codebook),
                static_cast<const float*>(residual_codebook) + residual_codebook_size);
        } else {
            residual_cb.centroids.assign(16, 0.0f);
        }
        residual_cb.boundaries.clear();
        residual_cb.origin = turboquant::CodebookOrigin::Empirical;

        // Wrap the packed primary / packed residual / norms buffers as
        // mlx::core::array instances. The array(T*, shape, dtype) ctor
        // copies the data into an MLX-owned buffer, so the caller's
        // pointer lifetime ends at this call site.
        auto primary_arr = mlx::core::array(
            static_cast<const uint8_t*>(packed_primary),
            {rank_out_features, primary_packed_cols},
            mlx::core::uint8);

        // Residual buffer: when absent, allocate a zero-filled array of
        // the expected shape so the kernel's has_residual detection loop
        // sees all zeros and takes the primary-only path.
        mlx::core::array residual_arr = [&]() {
            if (packed_residual && residual_bits > 0) {
                return mlx::core::array(
                    static_cast<const uint8_t*>(packed_residual),
                    {rank_out_features, residual_packed_cols},
                    mlx::core::uint8);
            }
            return mlx::core::zeros(
                {rank_out_features, residual_packed_cols},
                mlx::core::uint8);
        }();

        auto norms_arr = mlx::core::array(
            static_cast<const float*>(norms),
            {rank_out_features},
            mlx::core::float32);

        std::vector<uint32_t> seed_vec = {
            static_cast<uint32_t>(seed_primary),
            static_cast<uint32_t>(seed_residual),
            static_cast<uint32_t>(block_size),
        };
        auto seeds_arr = mlx::core::array(
            seed_vec.data(), {3}, mlx::core::uint32);

        turboquant::QuantizedWeight qw{
            std::move(primary_arr),
            std::move(residual_arr),
            std::move(norms_arr),
            std::move(seeds_arr),
        };

        auto handle = std::make_unique<tq_linear_s>();
        handle->layer = std::make_unique<turboquant::TurboQuantLinear>(
            local_in_features,
            rank_out_features,
            qw,
            primary_cb,
            residual_cb,
            static_cast<uint32_t>(block_size),
            full_in_features);
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

void* tq_linear_forward(tq_linear_t layer, const void* input_array) {
    if (!layer || !layer->layer || !input_array) return nullptr;
    try {
        const auto* input = static_cast<const mlx::core::array*>(input_array);
        auto result = layer->layer->forward(*input);
        return new mlx::core::array(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

void tq_linear_free(tq_linear_t layer) {
    if (!layer) return;
    delete layer;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

void tq_array_free(void* array) {
    if (!array) return;
    delete static_cast<mlx::core::array*>(array);
}

void* tq_array_from_fp32(const float* data, int rows, int cols) {
    try {
        if (!data || rows <= 0 || cols <= 0) return nullptr;
        auto arr = new mlx::core::array(
            data, {rows, cols}, mlx::core::float32);
        return arr;
    } catch (...) {
        return nullptr;
    }
}

// Implementation below.
int tq_array_copy_to_fp32(const void* array, float* out, int out_rows, int out_cols);

const char* tq_version(void) {
    return "0.1.0";
}

int tq_array_copy_to_fp32(const void* array, float* out, int out_rows, int out_cols) {
    if (!array || !out || out_rows <= 0 || out_cols <= 0) return -1;
    try {
        const auto* arr = static_cast<const mlx::core::array*>(array);
        if (arr->ndim() != 2) return -1;
        if (arr->shape(0) != out_rows || arr->shape(1) != out_cols) return -1;

        // Force graph materialisation so the buffer reflects the final
        // values; without it a just-dispatched Metal kernel's output
        // would not yet be resident on the CPU.
        mlx::core::array materialised = *arr;
        mlx::core::eval(materialised);
        auto f32 = mlx::core::astype(materialised, mlx::core::float32);
        mlx::core::eval(f32);
        const float* src = f32.data<float>();
        const size_t n = static_cast<size_t>(out_rows) * static_cast<size_t>(out_cols);
        for (size_t i = 0; i < n; ++i) out[i] = src[i];
        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"
