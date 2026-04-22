#include "turboquant/convert/shard_metadata.h"

#include <sstream>
#include <string>
#include <utility>

namespace turboquant::convert {

namespace {

/// Escape the minimum set of characters required for a valid JSON string
/// literal. Tensor names, strategy tags, and filenames are ASCII in practice,
/// but this guards against stray control characters corrupting the document.
std::string escape_json_string(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 2);
    for (char c : input) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Emit a JSON string literal including the enclosing quotes.
std::string json_quoted(const std::string& input) {
    return "\"" + escape_json_string(input) + "\"";
}

}  // namespace

void ShardMetadata::add_tensor(TensorEntry e) { tensors_.push_back(std::move(e)); }
void ShardMetadata::set_architecture(std::string a) { architecture_ = std::move(a); }
void ShardMetadata::set_hidden_size(int64_t h) { hidden_size_ = h; }
void ShardMetadata::set_num_attention_heads(int32_t n) { num_attention_heads_ = n; }
void ShardMetadata::set_intermediate_size(int64_t i) { intermediate_size_ = i; }
void ShardMetadata::set_num_experts(int32_t n) { num_experts_ = n; }
void ShardMetadata::set_top_k(int32_t k) { top_k_ = k; }

std::string ShardMetadata::to_json_string() const {
    // The document is small enough that building it with a string stream keeps
    // the implementation obvious and avoids pulling in a JSON dependency. The
    // convert tool writes the existing quantization_config block the same way.
    std::ostringstream os;

    os << "{\n";
    os << "  \"format_version\": " << FORMAT_VERSION << ",\n";
    os << "  \"model_architecture\": " << json_quoted(architecture_) << ",\n";
    os << "  \"hidden_size\": " << hidden_size_ << ",\n";
    os << "  \"num_attention_heads\": " << num_attention_heads_;
    if (intermediate_size_ > 0) {
        os << ",\n  \"intermediate_size\": " << intermediate_size_;
    }
    if (num_experts_ > 0) {
        os << ",\n  \"num_experts\": " << num_experts_;
        os << ",\n  \"top_k\": " << top_k_;
    }
    os << ",\n";

    os << "  \"tensors\": {";
    if (tensors_.empty()) {
        os << "}\n}\n";
        return os.str();
    }
    os << "\n";

    for (size_t i = 0; i < tensors_.size(); ++i) {
        const TensorEntry& t = tensors_[i];

        os << "    " << json_quoted(t.name) << ": {\n";

        // Shape array on a single line for readability.
        os << "      \"shape\": [";
        for (size_t j = 0; j < t.shape.size(); ++j) {
            if (j > 0) os << ", ";
            os << t.shape[j];
        }
        os << "],\n";

        os << "      \"dtype\": " << json_quoted(t.dtype) << ",\n";
        os << "      \"file\": " << json_quoted(t.file) << ",\n";
        os << "      \"byte_offset\": " << t.byte_offset << ",\n";
        os << "      \"byte_length\": " << t.byte_length << ",\n";

        // Distinguishing replicated (null) from an actual axis value is a
        // documented contract of the sidecar: SwiftLM uses this to decide
        // whether to broadcast the tensor or split it across ranks.
        if (t.shard_axis < 0) {
            os << "      \"shard_axis\": null,\n";
        } else {
            os << "      \"shard_axis\": " << t.shard_axis << ",\n";
        }

        os << "      \"shard_strategy\": " << json_quoted(t.shard_strategy);

        if (!t.codebook_key.empty()) {
            os << ",\n      \"codebook_key\": " << json_quoted(t.codebook_key);
        }
        if (!t.rotation_key.empty()) {
            os << ",\n      \"rotation_key\": " << json_quoted(t.rotation_key);
        }
        if (t.expert_index >= 0) {
            os << ",\n      \"expert_index\": " << t.expert_index;
        }

        os << "\n    }";
        if (i + 1 < tensors_.size()) os << ",";
        os << "\n";
    }

    os << "  }\n";
    os << "}\n";

    return os.str();
}

}  // namespace turboquant::convert
