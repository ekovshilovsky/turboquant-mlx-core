#pragma once

// Transport-agnostic serialized form of a TurboQuant KV cache. The same bytes
// are exchanged whether transferred over TCP, RDMA, or staged through disk.
// Per-layer payloads are opaque to this module; the TQKVCache owns their
// layout so wire-level changes do not leak the private storage schema.

#include <cstdint>
#include <vector>

namespace turboquant {

/// Complete serialized KV cache state for a single context.
struct TQCacheSnapshot {
    uint32_t num_layers = 0;
    uint32_t num_positions = 0;
    uint32_t head_dim = 0;
    uint32_t num_heads = 0;
    uint8_t  kv_bits = 3;
    uint32_t model_hash = 0;     ///< Identifier so the receiver can reject stale weights
    uint32_t codebook_hash = 0;  ///< Identifier for the active quantization codebook

    /// One opaque byte blob per layer. Length and internal layout are owned
    /// by TQKVCache; do not interpret outside the cache implementation.
    std::vector<std::vector<uint8_t>> layer_data;
};

/// Incremental cache update covering positions [from_position, to_position).
/// Used for background snapshots and live migration. The per-layer blobs use
/// the same internal layout as TQCacheSnapshot::layer_data so apply_delta
/// is a thin wrapper over the snapshot restore path.
struct TQCacheDelta {
    uint32_t from_position = 0;
    uint32_t to_position = 0;
    uint32_t num_layers = 0;

    std::vector<std::vector<uint8_t>> layer_data;
};

/// Serialize a snapshot into a single contiguous byte stream suitable for
/// network transfer. Layout is documented in cache_snapshot.cpp.
std::vector<uint8_t> snapshot_serialize(const TQCacheSnapshot& snap);

/// Reconstruct a snapshot from a byte stream produced by snapshot_serialize.
/// Returns an empty snapshot on malformed input; never throws.
TQCacheSnapshot snapshot_deserialize(const std::vector<uint8_t>& data);

/// Serialize a delta into a contiguous byte stream.
std::vector<uint8_t> delta_serialize(const TQCacheDelta& delta);

/// Reconstruct a delta from a byte stream. Returns an empty delta on
/// malformed input.
TQCacheDelta delta_deserialize(const std::vector<uint8_t>& data);

} // namespace turboquant
