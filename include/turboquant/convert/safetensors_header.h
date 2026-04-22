#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace turboquant::convert {

/// Parsed location record for a single tensor extracted from a safetensors
/// header. Byte offsets are absolute within the shard file (i.e. they already
/// include the fixed 8-byte prefix and the JSON header length).
struct SafetensorsHeaderEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t byte_offset = 0;
    int64_t byte_length = 0;
};

/// Parse the safetensors header of a shard and return a map from tensor name
/// to its on-disk location. The safetensors file layout is:
///
///   [8-byte little-endian header length] [JSON header] [raw tensor bytes]
///
/// Each tensor's "data_offsets" in the JSON header are relative to the start
/// of the tensor data region; the parser converts them to file-absolute
/// offsets so downstream code can seek directly into the shard.
///
/// The parser is intentionally narrow — it extracts only the fields needed
/// by the shard-metadata sidecar (dtype, shape, data_offsets) and does not
/// validate the full safetensors schema. Malformed or unreadable shards
/// return an empty map rather than raising an exception.
std::unordered_map<std::string, SafetensorsHeaderEntry>
parse_safetensors_header(const std::filesystem::path& shard_path);

}  // namespace turboquant::convert
