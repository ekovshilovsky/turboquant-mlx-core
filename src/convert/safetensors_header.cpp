#include "turboquant/convert/safetensors_header.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace turboquant::convert {

std::unordered_map<std::string, SafetensorsHeaderEntry>
parse_safetensors_header(const std::filesystem::path& shard_path) {
    std::unordered_map<std::string, SafetensorsHeaderEntry> out;

    std::ifstream in(shard_path, std::ios::binary);
    if (!in.is_open()) {
        return out;
    }

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    // The 4 GiB upper bound catches obviously garbage prefixes without
    // rejecting any legitimate safetensors header we are likely to encounter.
    if (!in || header_len == 0 || header_len > (1ull << 32)) {
        return out;
    }
    // Tensor data starts immediately after the fixed 8-byte prefix and the
    // variable-length JSON header.
    const int64_t data_region_start = static_cast<int64_t>(8 + header_len);

    std::string header(static_cast<size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        return out;
    }

    // Walk tensor entries by locating each top-level key. Each entry has the
    // shape { "dtype": "...", "shape": [...], "data_offsets": [start, end] }.
    // The __metadata__ key carries user metadata and is skipped.
    size_t pos = 0;
    while (pos < header.size()) {
        size_t key_start = header.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header.substr(key_start + 1, key_end - key_start - 1);
        pos = key_end + 1;

        // Skip anything that is not a tensor object — a tensor entry is the
        // key followed by a colon and an opening brace.
        size_t colon = header.find(':', pos);
        if (colon == std::string::npos) break;
        size_t brace = header.find_first_not_of(" \t\n\r", colon + 1);
        if (brace == std::string::npos || header[brace] != '{') {
            pos = colon + 1;
            continue;
        }

        // Locate the matching closing brace to bound the tensor entry substring.
        int depth = 1;
        size_t scan = brace + 1;
        while (scan < header.size() && depth > 0) {
            if (header[scan] == '{') ++depth;
            else if (header[scan] == '}') --depth;
            ++scan;
        }
        if (depth != 0) break;
        std::string entry = header.substr(brace, scan - brace);
        pos = scan;

        if (key == "__metadata__") continue;

        SafetensorsHeaderEntry e;

        // Extract dtype: first quoted string after the "dtype" key's colon.
        size_t dt = entry.find("\"dtype\"");
        if (dt != std::string::npos) {
            size_t colon_dt = entry.find(':', dt);
            if (colon_dt != std::string::npos) {
                size_t v1 = entry.find('"', colon_dt + 1);
                size_t v2 = (v1 != std::string::npos) ? entry.find('"', v1 + 1) : std::string::npos;
                if (v1 != std::string::npos && v2 != std::string::npos) {
                    e.dtype = entry.substr(v1 + 1, v2 - v1 - 1);
                }
            }
        }

        // Extract shape array.
        size_t sh = entry.find("\"shape\"");
        if (sh != std::string::npos) {
            size_t lb = entry.find('[', sh);
            size_t rb = (lb != std::string::npos) ? entry.find(']', lb) : std::string::npos;
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string nums = entry.substr(lb + 1, rb - lb - 1);
                size_t i = 0;
                while (i < nums.size()) {
                    while (i < nums.size() && (nums[i] == ' ' || nums[i] == ',')) ++i;
                    size_t start = i;
                    while (i < nums.size() && nums[i] != ',' && nums[i] != ' ') ++i;
                    if (i > start) {
                        try {
                            e.shape.push_back(std::stoll(nums.substr(start, i - start)));
                        } catch (...) {
                            // Malformed dimension — preserve the dimensions we
                            // could parse so the caller reports what it can.
                        }
                    }
                }
            }
        }

        // Extract data_offsets [start, end] and convert to absolute offsets.
        size_t off = entry.find("\"data_offsets\"");
        if (off != std::string::npos) {
            size_t lb = entry.find('[', off);
            size_t rb = (lb != std::string::npos) ? entry.find(']', lb) : std::string::npos;
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string nums = entry.substr(lb + 1, rb - lb - 1);
                size_t comma = nums.find(',');
                if (comma != std::string::npos) {
                    try {
                        int64_t rel_start = std::stoll(nums.substr(0, comma));
                        int64_t rel_end = std::stoll(nums.substr(comma + 1));
                        e.byte_offset = data_region_start + rel_start;
                        e.byte_length = rel_end - rel_start;
                    } catch (...) {
                        // Preserve zero offsets rather than propagating an
                        // exception through the caller.
                    }
                }
            }
        }

        out.emplace(std::move(key), std::move(e));
    }

    return out;
}

}  // namespace turboquant::convert
