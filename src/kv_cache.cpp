#include "turboquant/kv_cache.h"
#include "turboquant/quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace turboquant {

TQKVCache::TQKVCache(const KVCacheConfig& config)
    : config_(config),
      codebook_3bit_(generate_codebook(3)),
      layers_(static_cast<size_t>(config.num_layers)) {}

void TQKVCache::append(int layer, const mlx::core::array& keys,
                        const mlx::core::array& values) {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range("Layer index out of range");
    }

    // Expect keys/values shaped [num_tokens, num_heads, head_dim]
    mlx::core::eval(const_cast<mlx::core::array&>(keys));
    mlx::core::eval(const_cast<mlx::core::array&>(values));

    const int num_tokens = static_cast<int>(keys.shape(0));

    // Scramble layer+offset through a mixing function to avoid MT19937
    // short-range seed correlation between adjacent layers. Validated:
    // max cross-correlation stays within expected range for independent
    // random vectors (sqrt(2*ln(N))/sqrt(d) ~ 0.11 for N=27, d=512).
    auto scramble = [](uint32_t layer_idx, uint32_t offset) -> uint32_t {
        uint32_t h = layer_idx;
        h ^= offset + 0x9e3779b9u + (h << 6) + (h >> 2);
        h = (h ^ (h >> 16)) * 0x45d9f3bu;
        h = (h ^ (h >> 16)) * 0x45d9f3bu;
        h ^= h >> 16;
        return h;
    };
    uint32_t k_seed = scramble(static_cast<uint32_t>(layer), 1);
    uint32_t v_seed = scramble(static_cast<uint32_t>(layer), 2);
    uint32_t qjl_seed = scramble(static_cast<uint32_t>(layer), 3);

    auto& storage = layers_[static_cast<size_t>(layer)];

    // K-cache: 3-bit PolarQuant + 1-bit QJL residual for unbiased dot products
    quantize_and_store_with_qjl(keys, k_seed, qjl_seed, storage, num_tokens);
    // V-cache: 3-bit PolarQuant only (weighted-summed after softmax, no QJL needed)
    quantize_and_store(values, v_seed, storage.packed_value_indices,
                       storage.value_norms, num_tokens);

    storage.num_tokens += num_tokens;

    // Track global sequence length as the maximum across all layers,
    // since layers can be appended independently.
    int max_tokens = 0;
    for (const auto& ls : layers_) {
        if (ls.num_tokens > max_tokens) max_tokens = ls.num_tokens;
    }
    seq_length_ = max_tokens;
}

void TQKVCache::quantize_and_store(
    const mlx::core::array& data,
    uint32_t wht_seed,
    std::vector<uint32_t>& packed_indices,
    std::vector<float>& norms,
    int num_tokens) {

    const int num_heads = config_.num_heads;
    const int head_dim = config_.head_dim;
    const float scale = std::sqrt(static_cast<float>(head_dim));

    // Access raw float data — input is [num_tokens, num_heads, head_dim]
    auto data_f32 = mlx::core::astype(data, mlx::core::float32);
    mlx::core::eval(data_f32);
    const float* raw = data_f32.data<float>();

    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float* head_ptr = raw + (t * num_heads + h) * head_dim;

            // Compute per-head L2 norm
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum_sq += head_ptr[d] * head_ptr[d];
            }
            float norm = std::sqrt(sum_sq);
            norms.push_back(norm);

            // Normalize the head vector
            float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
            std::vector<float> normalized(static_cast<size_t>(head_dim));
            for (int d = 0; d < head_dim; ++d) {
                normalized[static_cast<size_t>(d)] = head_ptr[d] * inv_norm;
            }

            // WHT rotate: apply_wht_rotation expects [rows, cols] = [1, head_dim]
            auto norm_arr = mlx::core::array(
                normalized.data(), {1, head_dim}, mlx::core::float32);
            auto rotated = apply_wht_rotation(norm_arr, wht_seed,
                                              static_cast<uint32_t>(head_dim));
            mlx::core::eval(rotated);

            // Scale by sqrt(head_dim) for codebook compatibility
            const float* rot_ptr = rotated.data<float>();
            std::vector<float> scaled(static_cast<size_t>(head_dim));
            for (int d = 0; d < head_dim; ++d) {
                scaled[static_cast<size_t>(d)] = rot_ptr[d] * scale;
            }

            // Quantize to 3-bit indices via codebook
            auto scaled_arr = mlx::core::array(
                scaled.data(), {head_dim}, mlx::core::float32);
            auto indices = quantize(scaled_arr, codebook_3bit_);
            mlx::core::eval(indices);
            const uint8_t* idx_ptr = indices.data<uint8_t>();

            // Pack 3-bit indices: 10 per uint32 (30 bits used)
            for (int d = 0; d < head_dim; d += 10) {
                uint32_t word = 0;
                int remaining = std::min(10, head_dim - d);
                for (int j = 0; j < remaining; ++j) {
                    word |= (static_cast<uint32_t>(idx_ptr[d + j]) & 0x7)
                            << (j * 3);
                }
                packed_indices.push_back(word);
            }
        }
    }
}

void TQKVCache::quantize_and_store_with_qjl(
    const mlx::core::array& data,
    uint32_t wht_seed,
    uint32_t qjl_seed,
    LayerKVStorage& storage,
    int num_tokens) {

    const int num_heads = config_.num_heads;
    const int head_dim = config_.head_dim;
    const float scale = std::sqrt(static_cast<float>(head_dim));

    auto data_f32 = mlx::core::astype(data, mlx::core::float32);
    mlx::core::eval(data_f32);
    const float* raw = data_f32.data<float>();

    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float* head_ptr = raw + (t * num_heads + h) * head_dim;

            // Compute per-head L2 norm
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum_sq += head_ptr[d] * head_ptr[d];
            }
            float norm = std::sqrt(sum_sq);
            storage.key_norms.push_back(norm);

            // Normalize the head vector
            float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
            std::vector<float> normalized(static_cast<size_t>(head_dim));
            for (int d = 0; d < head_dim; ++d) {
                normalized[static_cast<size_t>(d)] = head_ptr[d] * inv_norm;
            }

            // WHT rotate: apply_wht_rotation expects [rows, cols] = [1, head_dim]
            auto norm_arr = mlx::core::array(
                normalized.data(), {1, head_dim}, mlx::core::float32);
            auto rotated = apply_wht_rotation(norm_arr, wht_seed,
                                              static_cast<uint32_t>(head_dim));
            mlx::core::eval(rotated);

            // Scale by sqrt(head_dim) for codebook compatibility
            const float* rot_ptr = rotated.data<float>();
            std::vector<float> scaled(static_cast<size_t>(head_dim));
            for (int d = 0; d < head_dim; ++d) {
                scaled[static_cast<size_t>(d)] = rot_ptr[d] * scale;
            }

            // Quantize to 3-bit indices via codebook
            auto scaled_arr = mlx::core::array(
                scaled.data(), {head_dim}, mlx::core::float32);
            auto indices = quantize(scaled_arr, codebook_3bit_);
            mlx::core::eval(indices);
            const uint8_t* idx_ptr = indices.data<uint8_t>();

            // Reconstruct quantized values in the scaled+rotated domain for
            // residual computation before packing
            std::vector<float> quantized_scaled(static_cast<size_t>(head_dim));
            const auto& centroids = codebook_3bit_.centroids;
            for (int d = 0; d < head_dim; ++d) {
                quantized_scaled[static_cast<size_t>(d)] = centroids[idx_ptr[d]];
            }

            // Compute QJL 1-bit residual correction in the scaled+rotated domain
            compute_qjl_residual(
                scaled.data(), quantized_scaled.data(), qjl_seed, head_dim,
                storage.key_qjl_signs, storage.key_qjl_norms);

            // Pack 3-bit indices: 10 per uint32 (30 bits used)
            for (int d = 0; d < head_dim; d += 10) {
                uint32_t word = 0;
                int remaining = std::min(10, head_dim - d);
                for (int j = 0; j < remaining; ++j) {
                    word |= (static_cast<uint32_t>(idx_ptr[d + j]) & 0x7)
                            << (j * 3);
                }
                storage.packed_key_indices.push_back(word);
            }
        }
    }
}

void TQKVCache::compute_qjl_residual(
    const float* rotated_scaled,
    const float* quantized_scaled,
    uint32_t qjl_seed,
    int head_dim,
    std::vector<uint32_t>& qjl_signs,
    std::vector<float>& qjl_norms) {

    // Compute residual vector in the scaled+rotated domain:
    // r[d] = original_scaled[d] - codebook_centroid[d]
    std::vector<float> residual(static_cast<size_t>(head_dim));
    float residual_sq = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        residual[static_cast<size_t>(d)] = rotated_scaled[d] - quantized_scaled[d];
        residual_sq += residual[static_cast<size_t>(d)] * residual[static_cast<size_t>(d)];
    }
    float qjl_norm = std::sqrt(residual_sq);
    qjl_norms.push_back(qjl_norm);

    // Project residual using deterministic random signs and store 1-bit
    // compressed projections. Each projection sign S[d] is derived from the
    // same hash used in Metal kernels for consistency:
    //   hash = qjl_seed * 2654435761 + d * 2246822519
    //   sign = (hash >> 31) ? +1 : -1
    // The projected value is: projected[d] = r[d] * sign[d]
    // We store sign(projected[d]) as a single bit.
    const int num_words = (head_dim + 31) / 32;
    for (int w = 0; w < num_words; ++w) {
        uint32_t sign_word = 0;
        int base = w * 32;
        int count = std::min(32, head_dim - base);
        for (int j = 0; j < count; ++j) {
            int d = base + j;
            // Deterministic random sign from seed, matching the Metal kernel hash
            uint32_t hash = qjl_seed * 2654435761u + static_cast<uint32_t>(d) * 2246822519u;
            float s = (hash >> 31) ? 1.0f : -1.0f;
            float projected = residual[static_cast<size_t>(d)] * s;
            if (projected > 0.0f) {
                sign_word |= (1u << j);
            }
        }
        qjl_signs.push_back(sign_word);
    }
}

mlx::core::array TQKVCache::get_keys_fp16(int layer, int start,
                                            int end) const {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range("Layer index out of range");
    }
    const auto& storage = layers_[static_cast<size_t>(layer)];
    auto scramble = [](uint32_t l, uint32_t o) -> uint32_t {
        uint32_t h = l; h ^= o + 0x9e3779b9u + (h << 6) + (h >> 2);
        h = (h ^ (h >> 16)) * 0x45d9f3bu; h = (h ^ (h >> 16)) * 0x45d9f3bu;
        return h ^ (h >> 16);
    };
    uint32_t k_seed = scramble(static_cast<uint32_t>(layer), 1);
    return dequantize_range(storage.packed_key_indices, storage.key_norms,
                            k_seed, storage.num_tokens, start, end);
}

mlx::core::array TQKVCache::get_values_fp16(int layer, int start,
                                              int end) const {
    if (layer < 0 || layer >= config_.num_layers) {
        throw std::out_of_range("Layer index out of range");
    }
    const auto& storage = layers_[static_cast<size_t>(layer)];
    auto scramble = [](uint32_t l, uint32_t o) -> uint32_t {
        uint32_t h = l; h ^= o + 0x9e3779b9u + (h << 6) + (h >> 2);
        h = (h ^ (h >> 16)) * 0x45d9f3bu; h = (h ^ (h >> 16)) * 0x45d9f3bu;
        return h ^ (h >> 16);
    };
    uint32_t v_seed = scramble(static_cast<uint32_t>(layer), 2);
    return dequantize_range(storage.packed_value_indices, storage.value_norms,
                            v_seed, storage.num_tokens, start, end);
}

mlx::core::array TQKVCache::dequantize_range(
    const std::vector<uint32_t>& packed_indices,
    const std::vector<float>& norms,
    uint32_t wht_seed,
    int total_tokens,
    int start,
    int end) const {

    const int num_heads = config_.num_heads;
    const int head_dim = config_.head_dim;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int num_tokens = end - start;

    if (start < 0 || end > total_tokens || start >= end) {
        throw std::out_of_range("Invalid token range for dequantization");
    }

    // Number of uint32 words per head (ceil(head_dim / 10))
    const int words_per_head = (head_dim + 9) / 10;
    // Number of uint32 words per token (all heads)
    const int words_per_token = words_per_head * num_heads;

    std::vector<float> output(
        static_cast<size_t>(num_tokens * num_heads * head_dim));
    const auto& centroids = codebook_3bit_.centroids;

    for (int t = 0; t < num_tokens; ++t) {
        int abs_token = start + t;
        for (int h = 0; h < num_heads; ++h) {
            // Locate packed data for this token and head
            int word_offset = abs_token * words_per_token
                              + h * words_per_head;

            // Unpack 3-bit indices and look up centroids
            std::vector<float> dequantized(static_cast<size_t>(head_dim));
            for (int d = 0; d < head_dim; ++d) {
                int word_idx = d / 10;
                int bit_pos = d % 10;
                uint32_t word =
                    packed_indices[static_cast<size_t>(word_offset + word_idx)];
                uint8_t idx = (word >> (bit_pos * 3)) & 0x7;
                dequantized[static_cast<size_t>(d)] = centroids[idx];
            }

            // Unscale
            for (int d = 0; d < head_dim; ++d) {
                dequantized[static_cast<size_t>(d)] *= inv_scale;
            }

            // Inverse WHT: expects [1, head_dim]
            auto dq_arr = mlx::core::array(
                dequantized.data(), {1, head_dim}, mlx::core::float32);
            auto inv_rot = apply_inverse_wht_rotation(
                dq_arr, wht_seed, static_cast<uint32_t>(head_dim));
            mlx::core::eval(inv_rot);
            const float* inv_ptr = inv_rot.data<float>();

            // Restore magnitude from stored norm
            float norm =
                norms[static_cast<size_t>(abs_token * num_heads + h)];
            size_t out_offset = static_cast<size_t>(
                (t * num_heads + h) * head_dim);
            for (int d = 0; d < head_dim; ++d) {
                output[out_offset + static_cast<size_t>(d)] =
                    inv_ptr[d] * norm;
            }
        }
    }

    auto result = mlx::core::array(
        output.data(), {num_tokens, num_heads, head_dim},
        mlx::core::float32);
    return mlx::core::astype(result, mlx::core::float16);
}

mlx::core::array TQKVCache::compressed_attention(
    int layer,
    const mlx::core::array& queries,
    int compressed_start,
    int compressed_end) const {
    // Fused TQ attention kernel is implemented in Task 5. Return an empty
    // array with the expected query-length dimension to maintain API shape.
    (void)layer;
    (void)compressed_start;
    (void)compressed_end;
    return mlx::core::array({});
}

int TQKVCache::seq_length() const { return seq_length_; }

void TQKVCache::clear() {
    seq_length_ = 0;
    for (auto& storage : layers_) {
        storage.packed_key_indices.clear();
        storage.packed_value_indices.clear();
        storage.key_norms.clear();
        storage.value_norms.clear();
        storage.key_qjl_signs.clear();
        storage.key_qjl_norms.clear();
        storage.num_tokens = 0;
    }
}

const KVCacheConfig& TQKVCache::config() const { return config_; }

// ---------------------------------------------------------------------------
// Snapshot + delta
// ---------------------------------------------------------------------------
//
// Per-layer blob layout (used for both full snapshots and deltas):
//   num_tokens (uint32)
//   key_indices_bytes (uint32) + raw uint32 packed key indices
//   value_indices_bytes (uint32) + raw uint32 packed value indices
//   key_norms_bytes (uint32) + raw float32 key norms
//   value_norms_bytes (uint32) + raw float32 value norms
//   qjl_signs_bytes (uint32) + raw uint32 QJL sign words
//   qjl_norms_bytes (uint32) + raw float32 QJL residual norms
//
// Because the append path writes one contiguous group of words per token
// (ceil(head_dim/10) key index words, num_heads norms, etc.), restricting a
// blob to positions [from, to) is a simple strided copy — no bit surgery.

namespace {

// Per-token stride in uint32 words for packed indices: ceil(head_dim/10) per head.
inline size_t packed_index_words_per_token(int num_heads, int head_dim) {
    return static_cast<size_t>(num_heads) * static_cast<size_t>((head_dim + 9) / 10);
}

// Per-token stride in uint32 words for QJL signs: ceil(head_dim/32) per head.
inline size_t qjl_sign_words_per_token(int num_heads, int head_dim) {
    return static_cast<size_t>(num_heads) * static_cast<size_t>((head_dim + 31) / 32);
}

// Per-token norm count: one float per head.
inline size_t norms_per_token(int num_heads) {
    return static_cast<size_t>(num_heads);
}

inline void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    const size_t pos = out.size();
    out.resize(pos + 4);
    std::memcpy(out.data() + pos, &v, 4);
}

template <typename T>
inline void append_span(std::vector<uint8_t>& out, const T* data, size_t count) {
    const uint32_t bytes = static_cast<uint32_t>(count * sizeof(T));
    append_u32(out, bytes);
    if (bytes == 0) return;
    const size_t pos = out.size();
    out.resize(pos + bytes);
    std::memcpy(out.data() + pos, data, bytes);
}

template <typename T>
inline bool read_span(const std::vector<uint8_t>& blob, size_t& pos,
                      std::vector<T>& dst, bool append_mode) {
    if (pos + 4 > blob.size()) return false;
    uint32_t bytes;
    std::memcpy(&bytes, blob.data() + pos, 4);
    pos += 4;
    if (pos + bytes > blob.size()) return false;

    const size_t count = bytes / sizeof(T);
    const size_t old_size = append_mode ? dst.size() : 0;
    dst.resize(old_size + count);
    if (bytes > 0) {
        std::memcpy(dst.data() + old_size, blob.data() + pos, bytes);
    }
    pos += bytes;
    return true;
}

/// Serialize tokens [from_tok, to_tok) of a single layer's storage. from_tok
/// and to_tok are clamped to the layer's token count to keep the format safe
/// against under-populated layers.
std::vector<uint8_t> serialize_layer_range(
    const LayerKVStorage& layer, int num_heads, int head_dim,
    int from_tok, int to_tok) {

    if (from_tok < 0) from_tok = 0;
    if (to_tok > layer.num_tokens) to_tok = layer.num_tokens;
    if (to_tok < from_tok) to_tok = from_tok;

    const size_t idx_stride = packed_index_words_per_token(num_heads, head_dim);
    const size_t qjl_stride = qjl_sign_words_per_token(num_heads, head_dim);
    const size_t norm_stride = norms_per_token(num_heads);
    const size_t n = static_cast<size_t>(to_tok - from_tok);

    const size_t idx_off = static_cast<size_t>(from_tok) * idx_stride;
    const size_t qjl_off = static_cast<size_t>(from_tok) * qjl_stride;
    const size_t norm_off = static_cast<size_t>(from_tok) * norm_stride;

    std::vector<uint8_t> blob;
    blob.reserve(24 + n * (2 * idx_stride + qjl_stride + 3 * norm_stride) * 4);

    append_u32(blob, static_cast<uint32_t>(n));
    append_span(blob, layer.packed_key_indices.data() + idx_off, n * idx_stride);
    append_span(blob, layer.packed_value_indices.data() + idx_off, n * idx_stride);
    append_span(blob, layer.key_norms.data() + norm_off, n * norm_stride);
    append_span(blob, layer.value_norms.data() + norm_off, n * norm_stride);
    append_span(blob, layer.key_qjl_signs.data() + qjl_off, n * qjl_stride);
    append_span(blob, layer.key_qjl_norms.data() + norm_off, n * norm_stride);
    return blob;
}

/// Load a per-layer blob into storage. On replace_mode=true the layer's
/// existing arrays are discarded first; otherwise the decoded spans are
/// appended to the current contents (used by apply_delta).
bool load_layer_blob(const std::vector<uint8_t>& blob, LayerKVStorage& layer,
                     bool replace_mode) {
    size_t pos = 0;
    if (pos + 4 > blob.size()) return false;
    uint32_t n;
    std::memcpy(&n, blob.data() + pos, 4);
    pos += 4;

    if (replace_mode) {
        layer.packed_key_indices.clear();
        layer.packed_value_indices.clear();
        layer.key_norms.clear();
        layer.value_norms.clear();
        layer.key_qjl_signs.clear();
        layer.key_qjl_norms.clear();
        layer.num_tokens = 0;
    }

    const bool append_mode = !replace_mode;
    if (!read_span(blob, pos, layer.packed_key_indices, append_mode)) return false;
    if (!read_span(blob, pos, layer.packed_value_indices, append_mode)) return false;
    if (!read_span(blob, pos, layer.key_norms, append_mode)) return false;
    if (!read_span(blob, pos, layer.value_norms, append_mode)) return false;
    if (!read_span(blob, pos, layer.key_qjl_signs, append_mode)) return false;
    if (!read_span(blob, pos, layer.key_qjl_norms, append_mode)) return false;

    layer.num_tokens += static_cast<int>(n);
    return true;
}

} // namespace

TQCacheSnapshot TQKVCache::snapshot() const {
    TQCacheSnapshot snap;
    snap.num_layers = static_cast<uint32_t>(config_.num_layers);
    snap.num_positions = static_cast<uint32_t>(seq_length_);
    snap.head_dim = static_cast<uint32_t>(config_.head_dim);
    snap.num_heads = static_cast<uint32_t>(config_.num_heads);
    snap.kv_bits = config_.k_bits;

    snap.layer_data.reserve(layers_.size());
    for (const auto& layer : layers_) {
        snap.layer_data.push_back(
            serialize_layer_range(layer, config_.num_heads, config_.head_dim,
                                  0, layer.num_tokens));
    }
    return snap;
}

void TQKVCache::restore(const TQCacheSnapshot& snap) {
    clear();

    const size_t n_layers = std::min<size_t>(layers_.size(), snap.layer_data.size());
    for (size_t i = 0; i < n_layers; ++i) {
        load_layer_blob(snap.layer_data[i], layers_[i], /*replace_mode=*/true);
    }

    int max_tokens = 0;
    for (const auto& l : layers_) {
        if (l.num_tokens > max_tokens) max_tokens = l.num_tokens;
    }
    seq_length_ = max_tokens;
}

TQCacheDelta TQKVCache::delta(uint32_t from, uint32_t to) const {
    TQCacheDelta d;
    d.from_position = from;
    d.to_position = to;
    d.num_layers = static_cast<uint32_t>(config_.num_layers);
    d.layer_data.reserve(layers_.size());

    const int from_int = static_cast<int>(from);
    const int to_int = static_cast<int>(to);
    for (const auto& layer : layers_) {
        d.layer_data.push_back(
            serialize_layer_range(layer, config_.num_heads, config_.head_dim,
                                  from_int, to_int));
    }
    return d;
}

void TQKVCache::apply_delta(const TQCacheDelta& d) {
    const size_t n_layers = std::min<size_t>(layers_.size(), d.layer_data.size());
    for (size_t i = 0; i < n_layers; ++i) {
        load_layer_blob(d.layer_data[i], layers_[i], /*replace_mode=*/false);
    }

    int max_tokens = 0;
    for (const auto& l : layers_) {
        if (l.num_tokens > max_tokens) max_tokens = l.num_tokens;
    }
    seq_length_ = max_tokens;
}

} // namespace turboquant
