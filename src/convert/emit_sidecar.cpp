#include "turboquant/convert/emit_sidecar.h"

#include "turboquant/convert/safetensors_header.h"
#include "turboquant/convert/shard_metadata.h"
#include "turboquant/convert/shard_strategy.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace turboquant::convert {

namespace {

/// Extract a single integer-valued top-level field from a HuggingFace
/// config.json. Looks for "field": <int> without recursing into nested
/// objects; sufficient for the dense scalar keys the shard sidecar exposes.
int64_t extract_config_int(const std::string& json, const std::string& field,
                           int64_t default_value) {
    const std::string key = "\"" + field + "\"";
    size_t pos = 0;
    while ((pos = json.find(key, pos)) != std::string::npos) {
        size_t colon = json.find(':', pos + key.size());
        if (colon == std::string::npos) return default_value;
        size_t i = colon + 1;
        while (i < json.size() &&
               (json[i] == ' ' || json[i] == '\t' || json[i] == '\n' || json[i] == '\r')) {
            ++i;
        }
        if (i < json.size() &&
            (std::isdigit(static_cast<unsigned char>(json[i])) || json[i] == '-')) {
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
std::string extract_config_string(const std::string& json, const std::string& field) {
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

/// TQ sub-tensor suffix check. The convert writer emits four tensors per
/// quantized weight (.packed_primary, .packed_residual, .norms, .seeds); the
/// sidecar exposes one logical entry per weight and rolls the companions
/// into codebook_key / rotation_key pointers.
bool is_tq_subtensor(const std::string& name) {
    return name.find(".packed_primary") != std::string::npos ||
           name.find(".packed_residual") != std::string::npos ||
           name.find(".norms") != std::string::npos ||
           name.find(".seeds") != std::string::npos;
}

}  // namespace

EmitSidecarResult emit_sidecar_for_directory(const fs::path& converted_dir) {
    EmitSidecarResult result;

    if (!fs::exists(converted_dir) || !fs::is_directory(converted_dir)) {
        result.error = "Input path does not exist or is not a directory: " +
                       converted_dir.string();
        return result;
    }

    const fs::path config_path = converted_dir / "config.json";
    if (!fs::exists(config_path)) {
        result.error = "No config.json found in " + converted_dir.string();
        return result;
    }

    // Enumerate safetensors shards up front so we can report a clear error
    // when the directory looks like a HuggingFace model but is missing
    // weight files entirely.
    std::vector<fs::path> shards;
    for (const auto& entry : fs::directory_iterator(converted_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path());
        }
    }
    if (shards.empty()) {
        result.error = "No *.safetensors files found in " + converted_dir.string();
        return result;
    }

    // Read model-level attributes from config.json. The quantization_config
    // block, if present, supplies the bits needed to reconstruct the tq{N}
    // dtype tag used for TQ-coded weights in the sidecar.
    std::ifstream cfg_in(config_path);
    const std::string config_json((std::istreambuf_iterator<char>(cfg_in)),
                                  std::istreambuf_iterator<char>());

    ShardMetadata md;

    // model_type is the short canonical form ("qwen2"); fall back to the
    // first entry of architectures ("Qwen2ForCausalLM") only if model_type
    // is missing — this keeps architecture strings stable across HF versions.
    std::string arch = extract_config_string(config_json, "model_type");
    if (arch.empty()) {
        arch = extract_config_string(config_json, "architectures");
    }
    md.set_architecture(arch);

    md.set_hidden_size(extract_config_int(config_json, "hidden_size", 0));
    md.set_num_attention_heads(
        static_cast<int32_t>(extract_config_int(config_json, "num_attention_heads", 0)));

    int64_t intermediate = extract_config_int(config_json, "intermediate_size", 0);
    if (intermediate > 0) md.set_intermediate_size(intermediate);

    int64_t num_experts = extract_config_int(config_json, "num_local_experts", 0);
    if (num_experts <= 0) num_experts = extract_config_int(config_json, "num_experts", 0);
    if (num_experts > 0) {
        md.set_num_experts(static_cast<int32_t>(num_experts));
        int64_t top_k = extract_config_int(config_json, "num_experts_per_tok", 0);
        if (top_k <= 0) top_k = extract_config_int(config_json, "top_k", 0);
        md.set_top_k(static_cast<int32_t>(top_k));
    }

    // Reconstruct the TQ dtype tag from the quantization_config block. The
    // block is injected by tq-convert; on a directory that predates that
    // field we fall back to "tq" so the tensor is still identifiable as
    // TurboQuant-coded without asserting a specific bit width.
    const int64_t primary_bits = extract_config_int(config_json, "bits", 0);
    const int64_t residual_bits = extract_config_int(config_json, "residual_bits", 0);
    std::string tq_dtype = "tq";
    if (primary_bits > 0) {
        tq_dtype = "tq" + std::to_string(primary_bits + residual_bits);
    }

    // Walk every safetensors shard and extract both TQ-coded logical weights
    // and passthrough tensors. The traversal mirrors the post-write pass in
    // the converter so the retrofit path and the inline path produce
    // byte-identical sidecars for the same on-disk layout.
    for (const auto& shard_path : shards) {
        const std::string shard_filename = shard_path.filename().string();
        auto entries = parse_safetensors_header(shard_path);

        // Group TQ sub-tensors by their logical base weight name. The sidecar
        // exposes one entry per logical weight, anchored at .packed_primary,
        // with codebook/rotation keys pointing to the companions.
        std::unordered_map<std::string, std::vector<std::string>> logical_groups;
        for (const auto& [name, _] : entries) {
            auto suffix_pos = name.find(".packed_primary");
            if (suffix_pos != std::string::npos) {
                std::string logical = name.substr(0, suffix_pos) + ".weight";
                logical_groups[logical].push_back(name);
            }
        }

        for (const auto& [logical_name, sub_names] : logical_groups) {
            const std::string base = logical_name.substr(0, logical_name.size() - 7);
            const auto primary_it = entries.find(base + ".packed_primary");
            if (primary_it == entries.end()) continue;

            TensorEntry te;
            te.name = logical_name;
            // Shape stored on disk is the packed shape [out, in/2]. Recover
            // the logical shape [out, in] so downstream loaders can reason
            // about the tensor the same way they do for fp16 weights.
            te.shape = primary_it->second.shape;
            if (te.shape.size() == 2) {
                te.shape[1] *= 2;
            }
            te.dtype = tq_dtype;
            te.file = shard_filename;
            te.byte_offset = primary_it->second.byte_offset;
            te.byte_length = primary_it->second.byte_length;
            te.shard_strategy = infer_shard_strategy(logical_name);
            te.shard_axis = infer_shard_axis(te.shard_strategy);
            te.codebook_key = base + ".codebook_primary";
            te.rotation_key = base + ".seeds";
            md.add_tensor(std::move(te));
        }

        // Emit entries for standalone tensors that are not TQ sub-tensors:
        // passthrough weights (embeddings, norms, biases, lm_head,
        // sensitive-layer fp16 weights) and codebook centroid tensors.
        for (const auto& [name, e] : entries) {
            if (is_tq_subtensor(name)) continue;

            TensorEntry te;
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

    const fs::path sidecar_path = converted_dir / "tq_shard_metadata.json";
    std::ofstream out(sidecar_path);
    if (!out.is_open()) {
        result.error = "Failed to open sidecar for writing: " + sidecar_path.string();
        return result;
    }
    out << md.to_json_string();
    out.close();
    if (!out) {
        result.error = "Failed to write sidecar: " + sidecar_path.string();
        return result;
    }

    // Summarize per-strategy counts for the CLI and tests. We re-read the
    // accumulated tensors from md would require an accessor; instead, we
    // recompute from the emitted JSON by re-walking the sidecar path would
    // be wasteful — count strategies by parsing the JSON once here. Cheaper:
    // iterate through the header parses again to tally. Cheapest: keep a
    // running tally while adding entries. We take the last option by
    // re-reading the freshly written JSON.
    //
    // Rationale: ShardMetadata deliberately does not expose its internal
    // vector (keeps the emitter-facing API tight). The freshly written
    // JSON is small (one entry per tensor, no payloads) so reading it back
    // once to tally strategies is cheap and avoids widening the class API
    // for a pure reporting concern.
    std::ifstream back(sidecar_path);
    const std::string sidecar_json((std::istreambuf_iterator<char>(back)),
                                   std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = sidecar_json.find("\"shard_strategy\"", pos)) != std::string::npos) {
        size_t q1 = sidecar_json.find('"', pos + 16);
        if (q1 == std::string::npos) break;
        size_t q2 = sidecar_json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string strategy = sidecar_json.substr(q1 + 1, q2 - q1 - 1);
        if (strategy == "column_parallel") ++result.column_parallel_count;
        else if (strategy == "row_parallel") ++result.row_parallel_count;
        else if (strategy == "replicated") ++result.replicated_count;
        else if (strategy == "expert_parallel") ++result.expert_parallel_count;
        ++result.tensor_count;
        pos = q2 + 1;
    }

    result.ok = true;
    result.sidecar_path = sidecar_path;
    return result;
}

}  // namespace turboquant::convert
