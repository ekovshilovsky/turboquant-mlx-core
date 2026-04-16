#include "turboquant_c/turboquant_c.h"
#include "turboquant/turboquant.h"
#include "turboquant/dequantizer.h"
#include "turboquant/kv_cache.h"
#include "turboquant/distributed.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

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
// Utility
// ---------------------------------------------------------------------------

void tq_array_free(void* array) {
    if (!array) return;
    delete static_cast<mlx::core::array*>(array);
}

const char* tq_version(void) {
    return "0.1.0";
}

} // extern "C"
