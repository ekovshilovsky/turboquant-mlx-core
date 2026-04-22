#include "turboquant/convert/emit_sidecar.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Lightweight substring check — keeps the test independent of a JSON parser
// while still validating every field the retrofit emitter must produce.
static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/// Create a minimal but valid safetensors file containing a fixed set of
/// tensors at the requested byte sizes. The returned vector records, for
/// each tensor, the absolute byte offset where its payload starts so the
/// test can cross-check the emitted sidecar against on-disk reality.
///
/// Safetensors layout: [u64 LE header_len][JSON header][raw tensor bytes].
struct WrittenTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t byte_offset = 0;  ///< File-absolute offset of the payload
    int64_t byte_length = 0;
};

static std::vector<WrittenTensor> write_minimal_safetensors(
    const fs::path& out_path,
    const std::vector<WrittenTensor>& requested) {

    // Compose the JSON header assigning each tensor a contiguous byte window
    // in the data region. The header references offsets relative to the
    // start of the data region; the file-absolute offset is (8 + header_len
    // + rel_offset), which we resolve after the header string is finalized.
    std::string header = "{";
    int64_t rel_cursor = 0;
    std::vector<std::pair<int64_t, int64_t>> rel_ranges;
    rel_ranges.reserve(requested.size());
    for (size_t i = 0; i < requested.size(); ++i) {
        const auto& t = requested[i];
        const int64_t start = rel_cursor;
        const int64_t end = start + t.byte_length;
        rel_ranges.emplace_back(start, end);
        rel_cursor = end;

        if (i > 0) header += ",";
        header += "\"" + t.name + "\":{";
        header += "\"dtype\":\"" + t.dtype + "\",";
        header += "\"shape\":[";
        for (size_t j = 0; j < t.shape.size(); ++j) {
            if (j > 0) header += ",";
            header += std::to_string(t.shape[j]);
        }
        header += "],";
        header += "\"data_offsets\":[" + std::to_string(start) + "," + std::to_string(end) + "]";
        header += "}";
    }
    header += "}";

    const uint64_t header_len = static_cast<uint64_t>(header.size());
    const int64_t data_region_start = static_cast<int64_t>(8 + header_len);

    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // Fill each tensor's byte window with deterministic bytes so the file is
    // a legitimate safetensors shard. The content does not matter for the
    // sidecar emitter — only the offsets do — but writing real bytes keeps
    // the file self-consistent for anything else that inspects it.
    std::mt19937 rng(0xC0DE);
    std::vector<uint8_t> buf;
    for (const auto& [start, end] : rel_ranges) {
        const int64_t len = end - start;
        buf.resize(static_cast<size_t>(len));
        for (auto& b : buf) b = static_cast<uint8_t>(rng() & 0xFF);
        out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(len));
    }
    out.close();

    // Populate absolute offsets for the caller to assert against.
    std::vector<WrittenTensor> written = requested;
    for (size_t i = 0; i < written.size(); ++i) {
        written[i].byte_offset = data_region_start + rel_ranges[i].first;
        written[i].byte_length = rel_ranges[i].second - rel_ranges[i].first;
    }
    return written;
}

/// Build a fake TQ-converted directory:
///   - config.json advertising a dense Qwen-style model at 8-bit TQ
///   - one safetensors file with a column-parallel, a row-parallel, and a
///     replicated tensor
static fs::path build_fixture_dir(const std::string& suffix) {
    const fs::path dir = fs::temp_directory_path() /
        ("tq_emit_sidecar_test_" + suffix);
    fs::remove_all(dir);
    fs::create_directories(dir);

    // Minimal HF-style config.json. Includes the quantization_config block
    // that tq-convert injects so the retrofit tool can synthesize the tq8
    // dtype tag from bits + residual_bits.
    const std::string config_json = R"({
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "intermediate_size": 11008,
  "num_hidden_layers": 2,
  "vocab_size": 151936,
  "quantization_config": {
    "quantization_method": "turboquant",
    "tq_version": "1",
    "bits": 4,
    "residual_bits": 4,
    "block_size": 512,
    "shared_rotation": false,
    "sensitive_layers_start": 0,
    "sensitive_layers_end": 0
  }
})";
    std::ofstream(dir / "config.json") << config_json;

    return dir;
}

/// The retrofit tool must produce a sidecar that records the right shard
/// strategy for each canonical tensor name and the correct byte offsets
/// relative to the on-disk safetensors file.
static void test_retrofit_emits_correct_strategies_and_offsets() {
    const fs::path dir = build_fixture_dir("strategies");

    // A column-parallel TQ-coded weight (.packed_primary sub-tensor anchor),
    // a row-parallel TQ-coded weight, and a replicated passthrough tensor.
    std::vector<WrittenTensor> requested = {
        {"model.layers.0.self_attn.q_proj.packed_primary", "uint8", {16, 16}, 0, 256},
        {"model.layers.0.self_attn.o_proj.packed_primary", "uint8", {16, 16}, 0, 256},
        {"model.embed_tokens.weight", "bfloat16", {128, 16}, 0, 4096},
    };
    const auto written = write_minimal_safetensors(
        dir / "model-00001-of-00001.safetensors", requested);

    const auto result = turboquant::convert::emit_sidecar_for_directory(dir);
    assert(result.ok);
    assert(result.sidecar_path == dir / "tq_shard_metadata.json");

    // Read the sidecar back and verify every field the loader depends on.
    std::ifstream in(result.sidecar_path);
    const std::string json_str((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());

    // Schema version and model-level attributes from config.json.
    assert(contains(json_str, "\"format_version\": 1"));
    assert(contains(json_str, "\"model_architecture\": \"qwen2\""));
    assert(contains(json_str, "\"hidden_size\": 2048"));
    assert(contains(json_str, "\"num_attention_heads\": 16"));
    assert(contains(json_str, "\"intermediate_size\": 11008"));

    // TQ-coded weights appear under their logical .weight name with tq8
    // dtype and an unpacked input dimension (safetensors shape [16, 16]
    // is the packed layout; the logical shape is [16, 32]).
    assert(contains(json_str, "\"model.layers.0.self_attn.q_proj.weight\""));
    assert(contains(json_str, "\"model.layers.0.self_attn.o_proj.weight\""));
    assert(contains(json_str, "\"dtype\": \"tq8\""));
    assert(contains(json_str, "\"shape\": [16, 32]"));

    // Strategy classification.
    assert(contains(json_str, "\"shard_strategy\": \"column_parallel\""));
    assert(contains(json_str, "\"shard_strategy\": \"row_parallel\""));
    assert(contains(json_str, "\"shard_strategy\": \"replicated\""));

    // Replicated tensors must declare a null shard axis so the loader
    // distinguishes "broadcast" from "split along axis 0".
    assert(contains(json_str, "\"shard_axis\": null"));

    // Codebook / rotation linkage for TQ-coded weights.
    assert(contains(json_str,
        "\"codebook_key\": \"model.layers.0.self_attn.q_proj.codebook_primary\""));
    assert(contains(json_str,
        "\"rotation_key\": \"model.layers.0.self_attn.q_proj.seeds\""));

    // Byte offsets in the sidecar must match on-disk reality exactly. The
    // replicated passthrough tensor's offset is the most informative check
    // because its logical entry in the sidecar keeps the original bytes.
    int64_t embed_offset = -1;
    int64_t embed_length = -1;
    for (const auto& w : written) {
        if (w.name == "model.embed_tokens.weight") {
            embed_offset = w.byte_offset;
            embed_length = w.byte_length;
        }
    }
    assert(embed_offset > 0);
    assert(contains(json_str, "\"byte_offset\": " + std::to_string(embed_offset)));
    assert(contains(json_str, "\"byte_length\": " + std::to_string(embed_length)));

    // Strategy counts reported through the result struct should match the
    // three-tensor fixture exactly (two TQ weights + one passthrough).
    assert(result.tensor_count == 3);
    assert(result.column_parallel_count == 1);
    assert(result.row_parallel_count == 1);
    assert(result.replicated_count == 1);
    assert(result.expert_parallel_count == 0);

    fs::remove_all(dir);
}

/// A directory without config.json is not a converted model — the retrofit
/// tool must fail cleanly rather than produce a half-populated sidecar.
static void test_missing_config_is_reported_as_error() {
    const fs::path dir = fs::temp_directory_path() / "tq_emit_sidecar_test_no_config";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const auto result = turboquant::convert::emit_sidecar_for_directory(dir);
    assert(!result.ok);
    assert(!result.error.empty());
    assert(!fs::exists(dir / "tq_shard_metadata.json"));

    fs::remove_all(dir);
}

/// A directory with a config.json but no safetensors is similarly invalid
/// — it may be a scratch directory, not a converted model.
static void test_missing_safetensors_is_reported_as_error() {
    const fs::path dir = fs::temp_directory_path() / "tq_emit_sidecar_test_no_shards";
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "config.json") << R"({"model_type": "qwen2"})";

    const auto result = turboquant::convert::emit_sidecar_for_directory(dir);
    assert(!result.ok);
    assert(!result.error.empty());
    assert(!fs::exists(dir / "tq_shard_metadata.json"));

    fs::remove_all(dir);
}

int main() {
    test_retrofit_emits_correct_strategies_and_offsets();
    printf("PASS: test_retrofit_emits_correct_strategies_and_offsets\n");

    test_missing_config_is_reported_as_error();
    printf("PASS: test_missing_config_is_reported_as_error\n");

    test_missing_safetensors_is_reported_as_error();
    printf("PASS: test_missing_safetensors_is_reported_as_error\n");

    return 0;
}
