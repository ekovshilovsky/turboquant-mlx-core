// Snapshot / delta serialization. The wire layout keeps TurboQuant's storage
// schema private to the cache while exposing a stable byte format that survives
// version upgrades of LayerKVStorage (new blob fields bump the format only).
//
// Snapshot layout:
//   magic (4 bytes)           : 'T','Q','C','S'
//   num_layers (uint32)       : count of per-layer blobs that follow
//   num_positions (uint32)    : max sequence length represented by the snapshot
//   head_dim (uint32)
//   num_heads (uint32)
//   kv_bits (uint8) + 3 pad   : quantization width at capture time
//   model_hash (uint32)       : identity of the weights that produced the cache
//   codebook_hash (uint32)    : identity of the active quantization codebook
//   for each layer:
//     blob_bytes (uint32)     : length of the opaque payload
//     blob (blob_bytes bytes) : per-layer TQKVCache-owned payload
//
// Delta layout mirrors the snapshot but replaces the cache-level header with
// from_position / to_position markers so receivers can reject out-of-order
// fragments without examining the payload.

#include "turboquant/cache_snapshot.h"

#include <cstring>

namespace turboquant {

namespace {

constexpr uint32_t kSnapshotMagic = 0x53435154u; // 'T','Q','C','S' little-endian
constexpr uint32_t kDeltaMagic    = 0x44435154u; // 'T','Q','C','D'

inline void write_u32(std::vector<uint8_t>& out, uint32_t v) {
    const size_t pos = out.size();
    out.resize(pos + 4);
    std::memcpy(out.data() + pos, &v, 4);
}

inline void write_blob(std::vector<uint8_t>& out, const std::vector<uint8_t>& blob) {
    write_u32(out, static_cast<uint32_t>(blob.size()));
    out.insert(out.end(), blob.begin(), blob.end());
}

inline bool read_u32(const std::vector<uint8_t>& data, size_t& pos, uint32_t& v) {
    if (pos + 4 > data.size()) return false;
    std::memcpy(&v, data.data() + pos, 4);
    pos += 4;
    return true;
}

inline bool read_blob(const std::vector<uint8_t>& data, size_t& pos,
                      std::vector<uint8_t>& blob) {
    uint32_t size;
    if (!read_u32(data, pos, size)) return false;
    if (pos + size > data.size()) return false;
    blob.assign(data.begin() + pos, data.begin() + pos + size);
    pos += size;
    return true;
}

} // namespace

std::vector<uint8_t> snapshot_serialize(const TQCacheSnapshot& snap) {
    std::vector<uint8_t> out;
    // Reserve the cache-level header plus a 4-byte length prefix for each
    // layer blob; actual payload size depends on the layers.
    out.reserve(32 + snap.layer_data.size() * 4);

    write_u32(out, kSnapshotMagic);
    write_u32(out, snap.num_layers);
    write_u32(out, snap.num_positions);
    write_u32(out, snap.head_dim);
    write_u32(out, snap.num_heads);
    uint32_t kv_bits_padded = snap.kv_bits; // upper 3 bytes zero-padded
    write_u32(out, kv_bits_padded);
    write_u32(out, snap.model_hash);
    write_u32(out, snap.codebook_hash);

    for (const auto& blob : snap.layer_data) {
        write_blob(out, blob);
    }
    return out;
}

TQCacheSnapshot snapshot_deserialize(const std::vector<uint8_t>& data) {
    TQCacheSnapshot snap;
    size_t pos = 0;

    uint32_t magic;
    if (!read_u32(data, pos, magic) || magic != kSnapshotMagic) return snap;
    if (!read_u32(data, pos, snap.num_layers)) return {};
    if (!read_u32(data, pos, snap.num_positions)) return {};
    if (!read_u32(data, pos, snap.head_dim)) return {};
    if (!read_u32(data, pos, snap.num_heads)) return {};

    uint32_t kv_bits_padded;
    if (!read_u32(data, pos, kv_bits_padded)) return {};
    snap.kv_bits = static_cast<uint8_t>(kv_bits_padded & 0xFFu);

    if (!read_u32(data, pos, snap.model_hash)) return {};
    if (!read_u32(data, pos, snap.codebook_hash)) return {};

    snap.layer_data.resize(snap.num_layers);
    for (uint32_t i = 0; i < snap.num_layers; ++i) {
        if (!read_blob(data, pos, snap.layer_data[i])) return {};
    }
    return snap;
}

std::vector<uint8_t> delta_serialize(const TQCacheDelta& delta) {
    std::vector<uint8_t> out;
    out.reserve(16 + delta.layer_data.size() * 4);

    write_u32(out, kDeltaMagic);
    write_u32(out, delta.from_position);
    write_u32(out, delta.to_position);
    write_u32(out, delta.num_layers);

    for (const auto& blob : delta.layer_data) {
        write_blob(out, blob);
    }
    return out;
}

TQCacheDelta delta_deserialize(const std::vector<uint8_t>& data) {
    TQCacheDelta delta;
    size_t pos = 0;

    uint32_t magic;
    if (!read_u32(data, pos, magic) || magic != kDeltaMagic) return delta;
    if (!read_u32(data, pos, delta.from_position)) return {};
    if (!read_u32(data, pos, delta.to_position)) return {};
    if (!read_u32(data, pos, delta.num_layers)) return {};

    delta.layer_data.resize(delta.num_layers);
    for (uint32_t i = 0; i < delta.num_layers; ++i) {
        if (!read_blob(data, pos, delta.layer_data[i])) return {};
    }
    return delta;
}

} // namespace turboquant
