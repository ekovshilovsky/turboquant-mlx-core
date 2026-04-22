#include "turboquant/convert/shard_metadata.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

// Lightweight substring check used as an assertion primitive. Keeps the test
// file independent of a JSON parser while still validating that every required
// field appears in the emitted document with the expected literal value.
static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// A column-parallel attention projection should emit every field required by
// SwiftLM's shard-aware weight loader: shape, dtype, byte window, shard axis,
// shard strategy, and codebook/rotation linkage.
static void test_emits_required_fields_for_column_parallel() {
    turboquant::convert::ShardMetadata md;
    turboquant::convert::TensorEntry entry;
    entry.name = "model.layers.0.self_attn.q_proj.weight";
    entry.shape = {3072, 2048};
    entry.dtype = "tq8";
    entry.file = "model-00001-of-00002.safetensors";
    entry.byte_offset = 98304;
    entry.byte_length = 6291456;
    entry.shard_axis = 0;
    entry.shard_strategy = "column_parallel";
    entry.codebook_key = "model.layers.0.self_attn.q_proj.codebook";
    entry.rotation_key = "model.layers.0.self_attn.q_proj.rotation";
    md.add_tensor(entry);
    md.set_architecture("qwen2");
    md.set_hidden_size(2048);
    md.set_num_attention_heads(16);

    const std::string json_str = md.to_json_string();

    assert(contains(json_str, "\"format_version\": 1"));
    assert(contains(json_str, "\"model_architecture\": \"qwen2\""));
    assert(contains(json_str, "\"hidden_size\": 2048"));
    assert(contains(json_str, "\"num_attention_heads\": 16"));
    assert(contains(json_str, "\"model.layers.0.self_attn.q_proj.weight\""));
    assert(contains(json_str, "\"shape\": [3072, 2048]"));
    assert(contains(json_str, "\"dtype\": \"tq8\""));
    assert(contains(json_str, "\"file\": \"model-00001-of-00002.safetensors\""));
    assert(contains(json_str, "\"byte_offset\": 98304"));
    assert(contains(json_str, "\"byte_length\": 6291456"));
    assert(contains(json_str, "\"shard_axis\": 0"));
    assert(contains(json_str, "\"shard_strategy\": \"column_parallel\""));
    assert(contains(json_str, "\"codebook_key\": \"model.layers.0.self_attn.q_proj.codebook\""));
    assert(contains(json_str, "\"rotation_key\": \"model.layers.0.self_attn.q_proj.rotation\""));
}

// A replicated tensor (e.g., token embedding) must serialize shard_axis as the
// JSON literal null so the loader can distinguish "replicate across ranks" from
// "shard along axis 0".
static void test_replicated_tensor_has_null_shard_axis() {
    turboquant::convert::ShardMetadata md;
    turboquant::convert::TensorEntry entry;
    entry.name = "model.embed_tokens.weight";
    entry.shape = {152064, 2048};
    entry.dtype = "tq8";
    entry.file = "model-00001-of-00002.safetensors";
    entry.byte_offset = 0;
    entry.byte_length = 311427072;
    entry.shard_axis = -1;
    entry.shard_strategy = "replicated";
    md.add_tensor(entry);

    const std::string json_str = md.to_json_string();
    assert(contains(json_str, "\"model.embed_tokens.weight\""));
    assert(contains(json_str, "\"shard_axis\": null"));
    assert(contains(json_str, "\"shard_strategy\": \"replicated\""));
}

// Optional fields (intermediate_size, num_experts, codebook_key, rotation_key,
// expert_index) must be omitted when unset so the schema stays minimal for the
// common dense case and grows only for MoE or residual-coded tensors.
static void test_optional_fields_omitted_when_unset() {
    turboquant::convert::ShardMetadata md;
    turboquant::convert::TensorEntry entry;
    entry.name = "model.norm.weight";
    entry.shape = {2048};
    entry.dtype = "f16";
    entry.file = "model-00001-of-00002.safetensors";
    entry.byte_offset = 0;
    entry.byte_length = 4096;
    entry.shard_axis = -1;
    entry.shard_strategy = "replicated";
    md.add_tensor(entry);
    md.set_architecture("qwen2");

    const std::string json_str = md.to_json_string();
    // No MoE configuration was supplied, so these keys must not appear
    assert(!contains(json_str, "intermediate_size"));
    assert(!contains(json_str, "num_experts"));
    assert(!contains(json_str, "top_k"));
    // No codebook/rotation linkage was supplied for this tensor
    assert(!contains(json_str, "codebook_key"));
    assert(!contains(json_str, "rotation_key"));
    assert(!contains(json_str, "expert_index"));
}

int main() {
    test_emits_required_fields_for_column_parallel();
    printf("PASS: test_emits_required_fields_for_column_parallel\n");

    test_replicated_tensor_has_null_shard_axis();
    printf("PASS: test_replicated_tensor_has_null_shard_axis\n");

    test_optional_fields_omitted_when_unset();
    printf("PASS: test_optional_fields_omitted_when_unset\n");

    return 0;
}
