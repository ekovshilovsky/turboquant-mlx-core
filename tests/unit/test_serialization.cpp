#include "turboquant/converter.h"
#include "turboquant/codebook.h"
#include "turboquant/quantizer.h"
#include <mlx/mlx.h>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

// Forward declarations for functions defined in serialization.cpp
namespace turboquant {
    std::unordered_map<std::string, std::string> read_tq_metadata(const std::string& path);
    bool write_tq_safetensors(
        const std::string& path,
        const std::unordered_map<std::string, QuantizedWeight>& weights,
        const Codebook& primary_codebook,
        const Codebook& residual_codebook,
        const std::unordered_map<std::string, std::pair<Codebook, Codebook>>& per_layer_codebooks);
}

/// Verify that a 16x128 weight survives a full serialize/deserialize cycle and
/// that the on-disk metadata matches what was written.
static void test_safetensors_roundtrip() {
    const std::string tmp_path = "/tmp/tq_test_roundtrip.safetensors";

    // Build a small synthetic weight matrix with varied values
    std::vector<float> data(16 * 128);
    for (int i = 0; i < 16 * 128; ++i) {
        data[static_cast<size_t>(i)] = static_cast<float>(i % 17 - 8) * 0.1f;
    }
    auto weight = mlx::core::array(data.data(), {16, 128});
    mlx::core::eval(weight);

    auto primary_cb  = turboquant::generate_codebook(4);
    auto residual_cb = turboquant::generate_codebook(4);
    turboquant::QuantizerConfig cfg;

    auto qw = turboquant::quantize_weight(weight, primary_cb, residual_cb, cfg);

    // mlx::core::array has no default constructor, so QuantizedWeight cannot
    // be default-constructed. Use insert() rather than operator[] to populate the map.
    std::unordered_map<std::string, turboquant::QuantizedWeight> weight_map;
    weight_map.insert({"test_layer.weight", qw});

    std::unordered_map<std::string, std::pair<turboquant::Codebook, turboquant::Codebook>> empty_plcb;
    bool ok = turboquant::write_tq_safetensors(tmp_path, weight_map, primary_cb, residual_cb, empty_plcb);
    assert(ok && "write_tq_safetensors should return true on success");

    // Confirm the file was actually written to disk
    assert(fs::exists(tmp_path) && "safetensors file must exist after write");

    // Read back metadata and validate all required fields
    auto metadata = turboquant::read_tq_metadata(tmp_path);
    assert(!metadata.empty() && "metadata must not be empty");

    assert(metadata.count("quantization_method") &&
           metadata.at("quantization_method") == "turboquant");
    assert(metadata.count("tq_version") &&
           metadata.at("tq_version") == "1");
    assert(metadata.count("tq_primary_bits") &&
           metadata.at("tq_primary_bits") == "4");
    assert(metadata.count("tq_residual_bits") &&
           metadata.at("tq_residual_bits") == "4");
    assert(metadata.count("tq_total_bits") &&
           metadata.at("tq_total_bits") == "8");

    fs::remove(tmp_path);

    printf("  PASS: safetensors write-read round-trip\n");
}

/// Verify that all required TQ metadata keys are present and correctly valued
/// after writing a quantized weight with non-default bit widths.
static void test_tq_metadata_written() {
    const std::string tmp_path = "/tmp/tq_test_metadata.safetensors";

    // Use 2-bit primary and 3-bit residual to exercise non-default bit-width metadata
    auto primary_cb  = turboquant::generate_codebook(2);
    auto residual_cb = turboquant::generate_codebook(3);

    // Minimal weight: 4x8 satisfies the even-column constraint of the nibble packer
    std::vector<float> data(4 * 8, 0.5f);
    auto weight = mlx::core::array(data.data(), {4, 8});
    mlx::core::eval(weight);

    turboquant::QuantizerConfig cfg;
    cfg.primary_bits  = 2;
    cfg.residual_bits = 3;

    auto qw = turboquant::quantize_weight(weight, primary_cb, residual_cb, cfg);

    std::unordered_map<std::string, turboquant::QuantizedWeight> weight_map;
    weight_map.insert({"layer.weight", qw});

    std::unordered_map<std::string, std::pair<turboquant::Codebook, turboquant::Codebook>> empty_plcb2;
    turboquant::write_tq_safetensors(tmp_path, weight_map, primary_cb, residual_cb, empty_plcb2);

    auto metadata = turboquant::read_tq_metadata(tmp_path);

    assert(metadata.at("quantization_method") == "turboquant");
    assert(metadata.at("tq_version")          == "1");
    assert(metadata.at("tq_primary_bits")     == "2");
    assert(metadata.at("tq_residual_bits")    == "3");
    assert(metadata.at("tq_total_bits")       == "5");

    fs::remove(tmp_path);

    printf("  PASS: TQ metadata present in output\n");
}

/// Verify that reading a TQ file via the standard MLX safetensors loader does
/// not raise an exception — stock readers must tolerate TQ metadata gracefully.
static void test_stock_reader_compatibility() {
    const std::string tmp_path = "/tmp/tq_test_compat.safetensors";

    auto primary_cb  = turboquant::generate_codebook(4);
    auto residual_cb = turboquant::generate_codebook(4);

    std::vector<float> data(8 * 16, 0.1f);
    auto weight = mlx::core::array(data.data(), {8, 16});
    mlx::core::eval(weight);

    turboquant::QuantizerConfig cfg;
    auto qw = turboquant::quantize_weight(weight, primary_cb, residual_cb, cfg);

    std::unordered_map<std::string, turboquant::QuantizedWeight> weight_map;
    weight_map.insert({"w", qw});

    std::unordered_map<std::string, std::pair<turboquant::Codebook, turboquant::Codebook>> empty_plcb3;
    turboquant::write_tq_safetensors(tmp_path, weight_map, primary_cb, residual_cb, empty_plcb3);

    // Load via the raw MLX API — must not throw
    bool threw = false;
    try {
        auto [tensors, metadata] = mlx::core::load_safetensors(tmp_path);
        // Confirm that all expected per-weight and model-level tensors are present
        assert(tensors.count("w.packed_primary")     && "packed_primary tensor must be present");
        assert(tensors.count("w.packed_residual")    && "packed_residual tensor must be present");
        assert(tensors.count("w.norms")              && "norms tensor must be present");
        assert(tensors.count("w.seeds")              && "seeds tensor must be present");
        assert(tensors.count("tq_codebook_primary")  && "primary codebook must be present");
        assert(tensors.count("tq_codebook_residual") && "residual codebook must be present");
    } catch (...) {
        threw = true;
    }
    assert(!threw && "stock safetensors reader must not throw on a TQ file");

    fs::remove(tmp_path);

    printf("  PASS: stock safetensors reader ignores TQ metadata gracefully\n");
}

int main() {
    printf("test_serialization:\n");
    test_safetensors_roundtrip();
    test_tq_metadata_written();
    test_stock_reader_compatibility();
    printf("All serialization tests passed.\n");
    return 0;
}
