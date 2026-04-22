#include "turboquant/converter.h"
#include "turboquant/codebook.h"
#include "turboquant/convert/shard_metadata.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace turboquant {

// Forward declarations for internal serialization functions defined in serialization.cpp.
// These are not exposed through a public header because they deal with TQ-specific
// tensor layouts that consumers should not depend on directly.
std::unordered_map<std::string, std::string> read_tq_metadata(const std::string& path);
bool write_tq_safetensors(
    const std::string& path,
    const std::unordered_map<std::string, QuantizedWeight>& weights,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    const std::unordered_map<std::string, std::pair<Codebook, Codebook>>& per_layer_codebooks);

/// Determine whether a tensor should be quantized.
/// Only 2D tensors whose name ends with ".weight" are candidates for quantization;
/// all other tensors (biases, embeddings, layer norms, etc.) are passed through unchanged.
static bool is_quantizable_weight(const std::string& name, const mlx::core::array& tensor) {
    if (tensor.ndim() != 2) {
        return false;
    }
    // Require the ".weight" suffix that identifies linear projection weights
    const std::string suffix = ".weight";
    if (name.size() < suffix.size()) {
        return false;
    }
    return name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Extract the numeric layer index from a tensor name following the standard
/// transformer naming conventions. Recognizes both "model.layers.N.*.weight"
/// and "model.language_model.layers.N.*.weight" patterns. Returns -1 for
/// tensors that are not part of a numbered layer (e.g., embed_tokens, lm_head).
static int extract_layer_index(const std::string& name) {
    static const std::regex layer_pattern(R"(\.layers\.(\d+)\.)");
    std::smatch match;
    if (std::regex_search(name, match, layer_pattern)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

/// Scan all tensor names across all shards to find the highest numbered layer.
/// Returns -1 if no numbered layers are found.
static int find_max_layer_index(const std::vector<fs::path>& shard_paths) {
    int max_index = -1;
    for (const auto& shard_path : shard_paths) {
        auto [tensors, metadata] = mlx::core::load_safetensors(shard_path.string());
        for (const auto& [name, tensor] : tensors) {
            int idx = extract_layer_index(name);
            if (idx > max_index) {
                max_index = idx;
            }
        }
    }
    return max_index;
}

/// Determine whether a layer should be kept at fp16 based on its position
/// relative to the sensitive layer boundaries. Layers at the beginning
/// (index < start) and end (index > max - end) of the transformer stack
/// are the most sensitive to quantization error.
static bool is_sensitive_layer(int layer_index, int max_layer_index,
                               int sensitive_start, int sensitive_end) {
    if (layer_index < 0) {
        return false;  // Not a numbered layer; sensitivity rules do not apply
    }
    if (layer_index < sensitive_start) {
        return true;
    }
    if (layer_index > max_layer_index - sensitive_end) {
        return true;
    }
    return false;
}

/// Infer the tensor-parallel shard strategy from a weight's fully qualified
/// name. Attention/MLP input projections shard along the output-feature axis
/// (column-parallel); attention/MLP output projections shard along the
/// input-feature axis (row-parallel). Everything else (embeddings, norms,
/// biases, lm_head) is treated as replicated.
static std::string infer_shard_strategy(const std::string& tensor_name) {
    if (tensor_name.find("q_proj.weight") != std::string::npos ||
        tensor_name.find("k_proj.weight") != std::string::npos ||
        tensor_name.find("v_proj.weight") != std::string::npos ||
        tensor_name.find("qkv_proj.weight") != std::string::npos ||
        tensor_name.find("gate_proj.weight") != std::string::npos ||
        tensor_name.find("up_proj.weight") != std::string::npos ||
        tensor_name.find("gate_up_proj.weight") != std::string::npos) {
        return "column_parallel";
    }
    if (tensor_name.find("o_proj.weight") != std::string::npos ||
        tensor_name.find("down_proj.weight") != std::string::npos) {
        return "row_parallel";
    }
    return "replicated";
}

/// Map a strategy tag to the canonical shard axis. Returns -1 for replicated
/// tensors to signal "no axis" to the metadata writer.
static int32_t infer_shard_axis(const std::string& strategy) {
    if (strategy == "column_parallel") return 0;
    if (strategy == "row_parallel") return 1;
    return -1;
}

/// Parsed offset entry for a single tensor extracted from a safetensors header.
struct SafetensorsHeaderEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t byte_offset = 0;   ///< Absolute byte offset within the shard file
    int64_t byte_length = 0;
};

/// Parse the safetensors header of a written shard and return a map from
/// tensor name to on-disk location. The safetensors format is:
///
///   [8-byte little-endian header length] [JSON header] [raw tensor bytes]
///
/// Each tensor's "data_offsets" in the JSON header is relative to the start of
/// the tensor data region, so we add the header prefix size to produce file-
/// absolute offsets. The parser is intentionally narrow — it only extracts the
/// fields the shard metadata needs and does not validate the full schema.
static std::unordered_map<std::string, SafetensorsHeaderEntry>
parse_safetensors_header(const fs::path& shard_path) {
    std::unordered_map<std::string, SafetensorsHeaderEntry> out;

    std::ifstream in(shard_path, std::ios::binary);
    if (!in.is_open()) {
        return out;
    }

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0 || header_len > (1ull << 32)) {
        return out;
    }
    // Data payload starts immediately after the fixed 8-byte prefix and the
    // variable-length JSON header
    const int64_t data_region_start = static_cast<int64_t>(8 + header_len);

    std::string header(static_cast<size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
        return out;
    }

    // Walk tensor entries by locating each top-level key. Each entry has the
    // shape { "dtype": "...", "shape": [...], "data_offsets": [start, end] }.
    // Safetensors writers (including MLX) emit the __metadata__ key for
    // user metadata — we skip it.
    size_t pos = 0;
    while (pos < header.size()) {
        size_t key_start = header.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header.substr(key_start + 1, key_end - key_start - 1);
        pos = key_end + 1;

        // Skip anything that is not a tensor object — must be followed by a
        // colon and an opening brace for an entry
        size_t colon = header.find(':', pos);
        if (colon == std::string::npos) break;
        size_t brace = header.find_first_not_of(" \t\n\r", colon + 1);
        if (brace == std::string::npos || header[brace] != '{') {
            pos = colon + 1;
            continue;
        }

        // Find the matching closing brace to bound the tensor entry substring
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

        // Extract dtype: find the colon after the key, then capture the first
        // quoted string after it. The key itself occupies seven characters
        // ("dtype" plus the enclosing quotes), so scanning past the key opener
        // is what matters — we look for the value's quotes directly.
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

        // Extract shape array
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
                            // Malformed shape entry — leave shape partial; the
                            // metadata writer reports what it can
                        }
                    }
                }
            }
        }

        // Extract data_offsets [start, end]
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
                        // exception through the convert tool
                    }
                }
            }
        }

        out.emplace(std::move(key), std::move(e));
    }

    return out;
}

/// Extract a single integer-valued top-level field from a HuggingFace
/// config.json. The parser is intentionally narrow: it looks for "field": <int>
/// without recursing into nested objects, which is sufficient for the dense
/// keys (hidden_size, num_attention_heads, intermediate_size, num_experts,
/// num_experts_per_tok) the sharding sidecar exposes.
static int64_t extract_config_int(const std::string& json, const std::string& field, int64_t default_value) {
    const std::string key = "\"" + field + "\"";
    size_t pos = 0;
    while ((pos = json.find(key, pos)) != std::string::npos) {
        size_t colon = json.find(':', pos + key.size());
        if (colon == std::string::npos) return default_value;
        size_t i = colon + 1;
        while (i < json.size() && (json[i] == ' ' || json[i] == '\t' || json[i] == '\n' || json[i] == '\r')) ++i;
        if (i < json.size() && (std::isdigit(static_cast<unsigned char>(json[i])) || json[i] == '-')) {
            size_t start = i;
            if (json[i] == '-') ++i;
            while (i < json.size() && std::isdigit(static_cast<unsigned char>(json[i]))) ++i;
            try {
                return std::stoll(json.substr(start, i - start));
            } catch (...) {
                return default_value;
            }
        }
        pos = colon + 1;
    }
    return default_value;
}

/// Extract a string-valued top-level field from a HuggingFace config.json.
static std::string extract_config_string(const std::string& json, const std::string& field) {
    const std::string key = "\"" + field + "\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return "";
    size_t colon = json.find(':', pos + key.size());
    if (colon == std::string::npos) return "";
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return json.substr(q1 + 1, q2 - q1 - 1);
}

/// Write tq_shard_metadata.json next to the converted safetensors files. The
/// sidecar records, for every logical weight tensor produced by the convert
/// pass, its shape, dtype, file location, byte window, and shard strategy, so
/// SwiftLM's shard-aware weight loader can dispatch TP shards without
/// re-parsing TurboQuant's packed on-disk layout.
static void write_shard_metadata_file(const ConversionConfig& config) {
    convert::ShardMetadata md;

    // Pull model-level attributes from the source config.json. The convert
    // tool already copies this file verbatim into the output directory, but
    // reading from the source keeps this helper independent of write order.
    const fs::path src_config_path = fs::path(config.input_path) / "config.json";
    if (fs::exists(src_config_path)) {
        std::ifstream in(src_config_path);
        const std::string json_str((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

        std::string arch = extract_config_string(json_str, "model_type");
        if (arch.empty()) arch = extract_config_string(json_str, "architectures");
        md.set_architecture(arch);

        md.set_hidden_size(extract_config_int(json_str, "hidden_size", 0));
        md.set_num_attention_heads(
            static_cast<int32_t>(extract_config_int(json_str, "num_attention_heads", 0)));

        int64_t intermediate = extract_config_int(json_str, "intermediate_size", 0);
        if (intermediate > 0) md.set_intermediate_size(intermediate);

        int64_t num_experts = extract_config_int(json_str, "num_local_experts", 0);
        if (num_experts <= 0) num_experts = extract_config_int(json_str, "num_experts", 0);
        if (num_experts > 0) {
            md.set_num_experts(static_cast<int32_t>(num_experts));
            int64_t top_k = extract_config_int(json_str, "num_experts_per_tok", 0);
            if (top_k <= 0) top_k = extract_config_int(json_str, "top_k", 0);
            md.set_top_k(static_cast<int32_t>(top_k));
        }
    }

    // Walk every safetensors file in the output directory and extract byte
    // offsets for each on-disk tensor. This accounts for both the main TQ
    // shards and the companion passthrough shards the convert pass writes
    // for tensors that are not quantized.
    std::vector<fs::path> output_shards;
    for (const auto& entry : fs::directory_iterator(config.output_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            output_shards.push_back(entry.path());
        }
    }

    for (const auto& shard_path : output_shards) {
        const std::string shard_filename = shard_path.filename().string();
        auto entries = parse_safetensors_header(shard_path);

        // Group TQ sub-tensors by their logical base weight name. The TQ
        // writer emits four tensors per quantized weight: .packed_primary,
        // .packed_residual, .norms, .seeds. The sidecar exposes one entry
        // per logical weight with the .packed_primary window as the anchor
        // and codebook/rotation keys pointing to the companions.
        std::unordered_map<std::string, std::vector<std::string>> logical_groups;
        for (const auto& [name, _] : entries) {
            auto suffix_pos = name.find(".packed_primary");
            if (suffix_pos != std::string::npos) {
                std::string logical = name.substr(0, suffix_pos) + ".weight";
                logical_groups[logical].push_back(name);
            }
        }

        // Emit one TensorEntry per logical quantized weight
        for (const auto& [logical_name, sub_names] : logical_groups) {
            const std::string base = logical_name.substr(0, logical_name.size() - 7);
            const auto primary_it = entries.find(base + ".packed_primary");
            if (primary_it == entries.end()) continue;

            convert::TensorEntry te;
            te.name = logical_name;
            // Shape stored in safetensors is the packed shape [out, in/2].
            // Recover the logical shape [out, in] so downstream loaders can
            // reason about the tensor the same way they do for fp16 weights.
            te.shape = primary_it->second.shape;
            if (te.shape.size() == 2) {
                te.shape[1] *= 2;
            }
            te.dtype = "tq" + std::to_string(config.quantizer.primary_bits +
                                             config.quantizer.residual_bits);
            te.file = shard_filename;
            te.byte_offset = primary_it->second.byte_offset;
            te.byte_length = primary_it->second.byte_length;
            te.shard_strategy = infer_shard_strategy(logical_name);
            te.shard_axis = infer_shard_axis(te.shard_strategy);
            te.codebook_key = base + ".codebook_primary";
            te.rotation_key = base + ".seeds";
            md.add_tensor(std::move(te));
        }

        // Emit entries for standalone tensors that are not TQ sub-tensors.
        // This covers passthrough weights (embeddings, norms, biases, lm_head,
        // sensitive-layer fp16 weights), model-level codebook centroids, and
        // per-layer codebook centroids. Downstream the loader can filter by
        // strategy or the "tq" dtype prefix as needed.
        auto is_tq_subtensor = [](const std::string& n) {
            return n.find(".packed_primary") != std::string::npos ||
                   n.find(".packed_residual") != std::string::npos ||
                   n.find(".norms") != std::string::npos ||
                   n.find(".seeds") != std::string::npos;
        };

        for (const auto& [name, e] : entries) {
            if (is_tq_subtensor(name)) continue;

            convert::TensorEntry te;
            te.name = name;
            te.shape = e.shape;
            te.dtype = e.dtype;
            te.file = shard_filename;
            te.byte_offset = e.byte_offset;
            te.byte_length = e.byte_length;
            te.shard_strategy = infer_shard_strategy(name);
            te.shard_axis = infer_shard_axis(te.shard_strategy);
            md.add_tensor(std::move(te));
        }
    }

    const fs::path metadata_path =
        fs::path(config.output_path) / "tq_shard_metadata.json";
    std::ofstream out(metadata_path);
    out << md.to_json_string();
}

bool convert_model(const ConversionConfig& config) {
    // Validate that the source model directory exists before doing any work
    if (!fs::exists(config.input_path) || !fs::is_directory(config.input_path)) {
        return false;
    }

    // Generate codebooks once; they are shared across all weight tensors in the model
    const Codebook primary_codebook  = generate_codebook(config.quantizer.primary_bits);
    const Codebook residual_codebook = generate_codebook(
        config.quantizer.residual_bits > 0 ? config.quantizer.residual_bits : 1);

    // Collect all safetensors shards present in the source directory
    std::vector<fs::path> safetensors_files;
    for (const auto& entry : fs::directory_iterator(config.input_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            safetensors_files.push_back(entry.path());
        }
    }

    // Determine the highest numbered layer across all shards for sensitive layer
    // boundary calculations. This pre-scan is necessary because shards may split
    // layers across files, so we need the global maximum before routing decisions.
    const bool has_sensitive_layers =
        config.quantizer.sensitive_layers_start > 0 ||
        config.quantizer.sensitive_layers_end > 0;
    const int max_layer_index = has_sensitive_layers
        ? find_max_layer_index(safetensors_files) : -1;

    // Ensure the output directory exists, creating it and any missing parents
    fs::create_directories(config.output_path);

    // Copy non-safetensors support files (config.json, tokenizer files, generation
    // configs, etc.) verbatim so the converted model directory remains complete
    for (const auto& entry : fs::directory_iterator(config.input_path)) {
        if (entry.is_regular_file() && entry.path().extension() != ".safetensors") {
            const fs::path dest = fs::path(config.output_path) / entry.path().filename();
            fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
        }
    }

    // Inject quantization_config into config.json so downstream loaders
    // (SwiftLM, HuggingFace transformers) can identify TQ-compressed models
    // without inspecting safetensors metadata. Follows the HuggingFace convention
    // where quantized models declare their method in the model config.
    {
        const fs::path config_path = fs::path(config.output_path) / "config.json";
        if (fs::exists(config_path)) {
            std::ifstream in(config_path);
            std::string json_str((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
            in.close();

            auto last_brace = json_str.rfind('}');
            if (last_brace != std::string::npos) {
                std::string quant_block =
                    ",\n  \"quantization_config\": {\n"
                    "    \"quantization_method\": \"turboquant\",\n"
                    "    \"tq_version\": \"1\",\n"
                    "    \"bits\": " + std::to_string(config.quantizer.primary_bits) + ",\n"
                    "    \"residual_bits\": " + std::to_string(config.quantizer.residual_bits) + ",\n"
                    "    \"block_size\": " + std::to_string(config.quantizer.block_size) + ",\n"
                    "    \"shared_rotation\": " + (config.quantizer.shared_rotation ? "true" : "false") + ",\n"
                    "    \"sensitive_layers_start\": " + std::to_string(config.quantizer.sensitive_layers_start) + ",\n"
                    "    \"sensitive_layers_end\": " + std::to_string(config.quantizer.sensitive_layers_end) + "\n"
                    "  }\n";

                json_str.replace(last_brace, 1, quant_block + "}");

                std::ofstream out(config_path);
                out << json_str;
            }
        }
    }

    // Count total weight tensors across all shards for accurate progress reporting.
    // This requires a preliminary pass over the files before the main conversion pass.
    int total_weights = 0;
    for (const auto& shard_path : safetensors_files) {
        auto [tensors, metadata] = mlx::core::load_safetensors(shard_path.string());
        for (const auto& [name, tensor] : tensors) {
            if (is_quantizable_weight(name, tensor)) {
                ++total_weights;
            }
        }
    }

    std::atomic<int> processed_count{0};
    int sensitive_kept_fp16 = 0;
    int layers_quantized = 0;

    // Main conversion pass: quantize eligible weights, write TQ safetensors per shard
    for (const auto& shard_path : safetensors_files) {
        auto [tensors, file_metadata] = mlx::core::load_safetensors(shard_path.string());

        // Partition tensors into those that will be quantized and those passed through
        std::unordered_map<std::string, QuantizedWeight> quantized_weights;
        std::unordered_map<std::string, mlx::core::array> passthrough_tensors;
        std::unordered_map<std::string, std::pair<Codebook, Codebook>> shard_layer_codebooks;

        // Collect all quantizable weights and their metadata before processing.
        // This two-pass approach separates the classification step (which must
        // be serial due to sensitive layer routing) from the quantization step
        // (which can be parallelized across independent weight tensors).
        struct WeightJob {
            std::string name;
            std::string layer_name;
            mlx::core::array fp32_tensor;
            QuantizerConfig layer_config;
        };
        std::vector<WeightJob> jobs;

        for (const auto& [name, tensor] : tensors) {
            if (is_quantizable_weight(name, tensor)) {
                // Check whether this weight belongs to a sensitive layer that
                // should be preserved at fp16 to minimize perplexity degradation
                if (has_sensitive_layers) {
                    int layer_idx = extract_layer_index(name);
                    if (is_sensitive_layer(layer_idx, max_layer_index,
                                           config.quantizer.sensitive_layers_start,
                                           config.quantizer.sensitive_layers_end)) {
                        passthrough_tensors.insert({name, tensor});
                        ++sensitive_kept_fp16;
                        continue;
                    }
                }

                // Strip the ".weight" suffix to form the canonical layer name used
                // as the key in the TQ per-weight tensor group (e.g., "layers.0.self_attn.q_proj")
                const std::string layer_name = name.substr(0, name.size() - 7);

                mlx::core::array fp32_tensor = mlx::core::astype(tensor, mlx::core::float32);
                mlx::core::eval(fp32_tensor);

                // Use the largest power-of-2 block_size that divides in_features
                // evenly. WHT requires full blocks — a remainder means unrotated
                // columns quantized with a codebook optimized for rotated
                // coordinates, producing catastrophic quality loss.
                QuantizerConfig layer_config = config.quantizer;
                int in_feat = static_cast<int>(fp32_tensor.shape(1));
                uint32_t best_bs = 1;
                for (int p = 1; p < 20; ++p) {
                    uint32_t c = 1u << p;
                    if (c > layer_config.block_size) break;
                    if (in_feat % static_cast<int>(c) == 0) best_bs = c;
                }
                layer_config.block_size = best_bs;

                jobs.push_back({name, layer_name, fp32_tensor, layer_config});
            } else {
                passthrough_tensors.insert({name, tensor});
            }
        }

        // Result type for collecting parallel quantization outputs. Each worker
        // returns the layer name, quantized weight, and optionally the per-layer
        // codebook pair when per_layer_codebooks is enabled. Uses optional to
        // avoid requiring a default constructor for QuantizedWeight (which
        // contains mlx::core::array members that have no default constructor).
        struct WeightResult {
            std::string layer_name;
            std::optional<QuantizedWeight> qw;
            bool has_per_layer_cb = false;
            Codebook layer_primary_cb;
            Codebook layer_residual_cb;
        };

        // Mutex protecting MLX operations that may not be thread-safe.
        // MLX's internal state (stream scheduling, device memory) is not
        // guaranteed thread-safe for concurrent eval() calls. All MLX array
        // evaluations and data pointer accesses are serialized through this
        // mutex, while the CPU-bound Lloyd-Max codebook fitting runs outside
        // the lock to maximize parallel throughput.
        std::mutex mlx_mutex;

        // Process all quantizable weights in the shard concurrently. Each weight
        // tensor is independent, so the only shared state is the MLX runtime
        // (protected by mlx_mutex) and the atomic progress counter.
        std::vector<std::future<WeightResult>> futures;
        futures.reserve(jobs.size());

        for (auto& job : jobs) {
            futures.push_back(std::async(std::launch::async,
                [&job, &config, &primary_codebook, &residual_codebook,
                 &mlx_mutex, &processed_count, total_weights]() -> WeightResult {

                // Force CPU-only execution in converter worker threads to
                // prevent Metal command buffer conflicts from concurrent
                // GPU dispatch across parallel workers
                set_force_cpu(true);

                WeightResult result;
                result.layer_name = job.layer_name;

                Codebook layer_primary_cb = primary_codebook;
                Codebook layer_residual_cb = residual_codebook;

                if (config.per_layer_codebooks) {
                    // Per-layer codebook fitting: extract the rotated+scaled weight
                    // distribution using MLX (serialized), then fit codebooks on CPU
                    // (parallelized). This split maximizes throughput because Lloyd-Max
                    // iteration is the dominant cost and runs lock-free.
                    int in_feat = static_cast<int>(job.fp32_tensor.shape(1));
                    float scale_factor = std::sqrt(static_cast<float>(in_feat));

                    std::vector<float> primary_samples;
                    std::vector<float> residual_samples;
                    bool need_residual = job.layer_config.residual_bits > 0;

                    {
                        std::lock_guard<std::mutex> lock(mlx_mutex);

                        // Normalize rows by L2 norm (matching quantizer.cpp pipeline)
                        auto sq = mlx::core::multiply(job.fp32_tensor, job.fp32_tensor);
                        auto row_sum = mlx::core::sum(sq, {1}, true);
                        auto row_norms = mlx::core::sqrt(row_sum);
                        auto safe_norms = mlx::core::maximum(row_norms, mlx::core::array(1e-10f));
                        auto w_norm = mlx::core::divide(job.fp32_tensor, safe_norms);
                        mlx::core::eval(w_norm);

                        // Derive the same deterministic seed the quantizer will use
                        // (FNV-1a hash of weight contents with salt=0)
                        uint32_t seed_primary_preview = 0;
                        {
                            const float* wdata = job.fp32_tensor.data<float>();
                            size_t n = static_cast<size_t>(job.fp32_tensor.size());
                            uint32_t hash = 2166136261u ^ 0u;
                            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(wdata);
                            size_t byte_count = n * sizeof(float);
                            size_t stride = (byte_count > 4096) ? (byte_count / 4096) : 1;
                            for (size_t i = 0; i < byte_count; i += stride) {
                                hash ^= bytes[i];
                                hash *= 16777619u;
                            }
                            seed_primary_preview = hash;
                        }

                        auto rotated = apply_wht_rotation(w_norm, seed_primary_preview, job.layer_config.block_size);
                        mlx::core::eval(rotated);
                        auto scaled = mlx::core::multiply(rotated, mlx::core::array(scale_factor));
                        mlx::core::eval(scaled);

                        // Copy rotated+scaled values to CPU vectors for lock-free
                        // codebook fitting. This avoids holding the MLX mutex during
                        // the expensive iterative Lloyd-Max convergence loop.
                        auto flat_scaled = mlx::core::reshape(scaled, {-1});
                        mlx::core::eval(flat_scaled);
                        const float* scaled_ptr = flat_scaled.data<float>();
                        size_t total_elems = static_cast<size_t>(flat_scaled.size());
                        primary_samples.assign(scaled_ptr, scaled_ptr + total_elems);

                        if (need_residual) {
                            // Pre-quantize with the shared codebook to extract residual
                            // distribution for fitting the residual codebook
                            auto indices_primary = turboquant::quantize(flat_scaled, primary_codebook);
                            mlx::core::eval(indices_primary);
                            auto dq_primary = turboquant::dequantize(indices_primary, primary_codebook);
                            mlx::core::eval(dq_primary);
                            auto residual_flat = mlx::core::subtract(flat_scaled, dq_primary);
                            mlx::core::eval(residual_flat);

                            const float* res_ptr = residual_flat.data<float>();
                            residual_samples.assign(res_ptr, res_ptr + total_elems);
                        }
                    }
                    // MLX mutex released — Lloyd-Max fitting runs in parallel across threads

                    layer_primary_cb = generate_codebook_from_data(
                        primary_samples, job.layer_config.primary_bits, 100);

                    if (need_residual) {
                        layer_residual_cb = generate_codebook_from_data(
                            residual_samples, job.layer_config.residual_bits, 100);
                    }

                    result.has_per_layer_cb = true;
                    result.layer_primary_cb = layer_primary_cb;
                    result.layer_residual_cb = layer_residual_cb;
                }

                // Quantize the weight tensor through the full TurboQuant pipeline.
                // This call involves MLX operations internally, so it must be
                // serialized. For models without per-layer codebooks, this is the
                // only significant work per weight — parallelism still helps by
                // overlapping the CPU-heavy WHT and packing stages across weights.
                {
                    std::lock_guard<std::mutex> lock(mlx_mutex);
                    result.qw = quantize_weight(job.fp32_tensor, layer_primary_cb,
                                                layer_residual_cb, job.layer_config);
                }

                // Report progress with an atomic counter for thread-safe updates
                int current = ++processed_count;
                if (config.progress_callback) {
                    config.progress_callback(current, total_weights, job.name);
                }

                return result;
            }));
        }

        // Collect results from all parallel workers and merge into the shard's
        // output maps. Futures are consumed in submission order; each .get()
        // blocks until its worker completes, propagating any exceptions.
        for (auto& f : futures) {
            auto result = f.get();
            quantized_weights.insert({result.layer_name, std::move(*result.qw)});
            if (result.has_per_layer_cb) {
                shard_layer_codebooks.insert(
                    {result.layer_name,
                     {result.layer_primary_cb, result.layer_residual_cb}});
            }
            ++layers_quantized;
        }

        // Derive output shard path preserving the original filename
        const fs::path out_shard = fs::path(config.output_path) / shard_path.filename();

        // Write quantized weights together with codebooks and TQ metadata
        write_tq_safetensors(
            out_shard.string(), quantized_weights, primary_codebook, residual_codebook,
            shard_layer_codebooks);

        // Write passthrough tensors to a companion shard so no tensors are dropped
        if (!passthrough_tensors.empty()) {
            const std::string passthrough_path =
                (fs::path(config.output_path) /
                 (shard_path.stem().string() + "_passthrough.safetensors")).string();
            mlx::core::save_safetensors(passthrough_path, passthrough_tensors);
        }
    }

    // Report sensitive layer routing statistics when mixed precision is active
    if (has_sensitive_layers) {
        printf("Sensitive layers: %d weights kept at fp16, %d weights quantized\n",
               sensitive_kept_fp16, layers_quantized);
    }

    // Emit the tq_shard_metadata.json sidecar consumed by SwiftLM's
    // shard-aware weight loader. This is a pure post-write pass — it only
    // reads the safetensors headers already persisted above.
    write_shard_metadata_file(config);

    return true;
}

bool validate_converted_model(const std::string& model_path) {
    if (!fs::exists(model_path) || !fs::is_directory(model_path)) {
        return false;
    }

    // Scan all safetensors files in the output directory looking for TQ metadata.
    // A valid conversion must have at least one shard tagged with the TurboQuant marker.
    for (const auto& entry : fs::directory_iterator(model_path)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".safetensors") continue;

        auto metadata = read_tq_metadata(entry.path().string());
        auto it = metadata.find("quantization_method");
        if (it != metadata.end() && it->second == "turboquant") {
            return true;
        }
    }

    return false;
}

} // namespace turboquant
