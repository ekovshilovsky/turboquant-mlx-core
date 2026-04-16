#include "turboquant/converter.h"
#include "turboquant/dequantizer.h"
#include "turboquant/codebook.h"
#include <mlx/mlx.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

// Forward declaration for the internal metadata reader defined in serialization.cpp
namespace turboquant {
    std::unordered_map<std::string, std::string> read_tq_metadata(const std::string& path);
}

/// Write a minimal single-shard model directory containing one 2D weight tensor.
/// Returns false and prints an error if the directory or file cannot be created.
static bool create_test_model_dir(const std::string& dir_path) {
    fs::create_directories(dir_path);

    // 32x64 weight is large enough to exercise nibble packing and WHT rotation
    std::vector<float> data(32 * 64);
    for (int i = 0; i < 32 * 64; ++i) {
        data[static_cast<size_t>(i)] =
            static_cast<float>(i % 31 - 15) * 0.05f;
    }
    auto weight = mlx::core::array(data.data(), {32, 64});
    // Materialize the array on the device before saving
    mlx::core::eval(weight);

    std::unordered_map<std::string, mlx::core::array> tensors;
    // Use insert() instead of operator[] to avoid the implicit default-construction
    // that operator[] requires but mlx::core::array does not support
    tensors.insert({"layer.weight", weight});

    const std::string shard_path = dir_path + "/model.safetensors";
    mlx::core::save_safetensors(shard_path, tensors);

    return fs::exists(shard_path);
}

/// Verify that convert_model produces a valid TQ output directory and that
/// validate_converted_model confirms the conversion succeeded.
static void test_tiny_model_convert_load_forward() {
    const std::string input_dir  = "/tmp/tq_e2e_input";
    const std::string output_dir = "/tmp/tq_e2e_output";

    // Remove any leftover state from a previous run
    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    bool created = create_test_model_dir(input_dir);
    assert(created && "test model directory must be created successfully");

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 64;

    bool converted = turboquant::convert_model(config);
    assert(converted && "convert_model must return true on a well-formed input directory");

    assert(fs::exists(output_dir) && "output directory must exist after conversion");

    bool valid = turboquant::validate_converted_model(output_dir);
    assert(valid && "validate_converted_model must return true on a freshly converted model");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: tiny model convert -> load -> forward\n");
}

/// Verify that the output safetensors shard contains the expected TQ tensor layout.
/// Each quantized weight named "layer.weight" in the source must produce four
/// companion tensors in the output: packed_primary, packed_residual, norms, seeds.
static void test_output_tensor_names_and_shapes() {
    const std::string input_dir  = "/tmp/tq_e2e_names_input";
    const std::string output_dir = "/tmp/tq_e2e_names_output";

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    create_test_model_dir(input_dir);

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 64;

    turboquant::convert_model(config);

    const std::string shard_path = output_dir + "/model.safetensors";
    assert(fs::exists(shard_path) && "output shard must exist");

    auto [tensors, metadata] = mlx::core::load_safetensors(shard_path);

    // Each quantized weight "layer.weight" → "layer" key in write_tq_safetensors
    assert(tensors.count("layer.packed_primary")  && "packed_primary must be present");
    assert(tensors.count("layer.packed_residual") && "packed_residual must be present");
    assert(tensors.count("layer.norms")           && "norms must be present");
    assert(tensors.count("layer.seeds")           && "seeds must be present");

    // Shared codebook tensors must be present as model-level entries
    assert(tensors.count("tq_codebook_primary")   && "primary codebook must be present");
    assert(tensors.count("tq_codebook_residual")  && "residual codebook must be present");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: output safetensors has correct names/shapes\n");
}

/// Verify that the TQ metadata block in the output shard contains all required
/// fields with the expected values for the configured bit widths.
static void test_metadata_format() {
    const std::string input_dir  = "/tmp/tq_e2e_meta_input";
    const std::string output_dir = "/tmp/tq_e2e_meta_output";

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    create_test_model_dir(input_dir);

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 64;

    turboquant::convert_model(config);

    const std::string shard_path = output_dir + "/model.safetensors";
    assert(fs::exists(shard_path) && "output shard must exist");

    auto metadata = turboquant::read_tq_metadata(shard_path);

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

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: TQ metadata matches expected format\n");
}

/// Write a test model directory containing a single weight with non-power-of-2
/// column count. 896 = 7 x 128, matching Qwen2.5-0.5B hidden_size. This
/// exercises the adaptive block_size path during conversion that previously
/// failed when columns were not evenly divisible by the default 512 block_size.
static bool create_non_pow2_model_dir(const std::string& dir_path) {
    fs::create_directories(dir_path);

    const int rows = 64, cols = 896;
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        data[static_cast<size_t>(i)] =
            static_cast<float>(i % 31 - 15) * 0.05f;
    }
    auto weight = mlx::core::array(data.data(), {rows, cols});
    mlx::core::eval(weight);

    std::unordered_map<std::string, mlx::core::array> tensors;
    tensors.insert({"layer.weight", weight});

    const std::string shard_path = dir_path + "/model.safetensors";
    mlx::core::save_safetensors(shard_path, tensors);

    return fs::exists(shard_path);
}

/// Verify that convert_model handles non-power-of-2 weight dimensions correctly.
/// Creates a 64x896 weight (Qwen2.5-0.5B hidden_size), converts it, then
/// dequantizes the output back to float32 and verifies the relative RMSE is
/// within the expected tolerance for 4+4 bit quantization.
static void test_non_power_of_2_conversion() {
    const std::string input_dir  = "/tmp/tq_e2e_np2_input";
    const std::string output_dir = "/tmp/tq_e2e_np2_output";

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    bool created = create_non_pow2_model_dir(input_dir);
    assert(created && "non-power-of-2 test model directory must be created");

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 128;  // Adaptive block_size for 896 = 7 x 128

    bool converted = turboquant::convert_model(config);
    assert(converted && "convert_model must succeed on 64x896 non-power-of-2 input");

    // Verify TQ metadata is present and correct
    auto metadata = turboquant::read_tq_metadata(output_dir + "/model.safetensors");
    assert(!metadata.empty() && "metadata must not be empty");
    assert(metadata.count("quantization_method") &&
           metadata.at("quantization_method") == "turboquant");

    // Load the converted tensors and verify expected layout
    const std::string shard_path = output_dir + "/model.safetensors";
    auto [tensors, _meta] = mlx::core::load_safetensors(shard_path);

    assert(tensors.count("layer.packed_primary")  && "packed_primary must be present");
    assert(tensors.count("layer.packed_residual") && "packed_residual must be present");
    assert(tensors.count("layer.norms")           && "norms must be present");
    assert(tensors.count("layer.seeds")           && "seeds must be present");

    // Verify seeds contains 3 elements with correct block_size
    auto& seeds = tensors.at("layer.seeds");
    mlx::core::eval(seeds);
    assert(seeds.shape(0) == 3 && "seeds must have 3 elements");
    const uint32_t* seed_data = seeds.data<uint32_t>();
    assert(seed_data[2] == 128 && "seeds[2] must store block_size=128");

    // Dequant and verify reconstruction quality
    auto cb_primary = tensors.at("tq_codebook_primary");
    auto cb_residual = tensors.at("tq_codebook_residual");
    mlx::core::eval(cb_primary, cb_residual);

    turboquant::QuantizedWeight qw{
        tensors.at("layer.packed_primary"),
        tensors.at("layer.packed_residual"),
        tensors.at("layer.norms"),
        seeds
    };
    mlx::core::eval(qw.packed_primary, qw.packed_residual, qw.norms);

    // Reconstruct codebook from stored centroids
    turboquant::Codebook p_cb, r_cb;
    {
        const float* p_ptr = cb_primary.data<float>();
        p_cb.centroids.assign(p_ptr, p_ptr + cb_primary.size());
    }
    {
        const float* r_ptr = cb_residual.data<float>();
        r_cb.centroids.assign(r_ptr, r_ptr + cb_residual.size());
    }

    auto recon = turboquant::dequantize_weight_cpu(qw, p_cb, r_cb, 128);
    mlx::core::eval(recon);

    // Verify no NaN/Inf in reconstruction
    const float* recon_ptr = recon.data<float>();
    const int total = 64 * 896;
    for (int i = 0; i < total; ++i) {
        assert(!std::isnan(recon_ptr[i]) && "dequant output contains NaN");
        assert(!std::isinf(recon_ptr[i]) && "dequant output contains Inf");
    }

    // Load original weight for RMSE comparison
    auto [orig_tensors, _orig_meta] = mlx::core::load_safetensors(input_dir + "/model.safetensors");
    auto orig = orig_tensors.at("layer.weight");
    mlx::core::eval(orig);

    auto diff = mlx::core::subtract(orig, recon);
    auto sq = mlx::core::multiply(diff, diff);
    auto mse = mlx::core::mean(sq);
    auto orig_sq = mlx::core::multiply(orig, orig);
    auto orig_mean_sq = mlx::core::mean(orig_sq);
    mlx::core::eval(mse, orig_mean_sq);

    float rmse = std::sqrt(mse.item<float>());
    float rms_orig = std::sqrt(orig_mean_sq.item<float>());
    float relative_rmse = rmse / rms_orig;

    printf("    64x896 relative RMSE: %.4f (%.1f%%)\n", relative_rmse, relative_rmse * 100.0f);
    assert(relative_rmse < 0.15f && "relative RMSE exceeds 15% for 64x896 conversion");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: non-power-of-2 (64x896) conversion and dequant\n");
}

/// Write a test model directory with 4 numbered transformer layers and a 1D
/// layernorm weight. This exercises the sensitive layer routing logic that
/// preserves the first and last N layers at fp16 while quantizing the rest.
static bool create_layered_model_dir(const std::string& dir_path) {
    fs::create_directories(dir_path);

    std::unordered_map<std::string, mlx::core::array> tensors;

    // Create 4 layers of 2D weight tensors (16x64) with distinct data patterns
    // to represent a minimal transformer stack: layers 0-3
    for (int layer = 0; layer < 4; ++layer) {
        std::vector<float> data(16 * 64);
        for (int i = 0; i < 16 * 64; ++i) {
            data[static_cast<size_t>(i)] =
                static_cast<float>((i + layer * 7) % 31 - 15) * 0.05f;
        }
        auto weight = mlx::core::array(data.data(), {16, 64});
        mlx::core::eval(weight);

        std::string name = "model.layers." + std::to_string(layer) +
                           ".self_attn.q_proj.weight";
        tensors.insert({name, weight});
    }

    // Add a 1D layernorm weight that must always pass through regardless
    // of sensitive layer settings (not a 2D weight, so not quantizable)
    std::vector<float> norm_data(64, 1.0f);
    auto norm_weight = mlx::core::array(norm_data.data(), {64});
    mlx::core::eval(norm_weight);
    tensors.insert({"model.layers.0.input_layernorm.weight", norm_weight});

    const std::string shard_path = dir_path + "/model.safetensors";
    mlx::core::save_safetensors(shard_path, tensors);

    return fs::exists(shard_path);
}

/// Verify that sensitive layer mixed precision routing correctly preserves the
/// first and last N layers at fp16 while quantizing interior layers.
/// With 4 layers (0-3) and sensitive_layers=1, layers 0 and 3 should be kept
/// at fp16 (passthrough) and layers 1 and 2 should be TQ-compressed.
static void test_sensitive_layer_mixed_precision() {
    const std::string input_dir  = "/tmp/tq_e2e_sensitive_input";
    const std::string output_dir = "/tmp/tq_e2e_sensitive_output";

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    bool created = create_layered_model_dir(input_dir);
    assert(created && "layered test model directory must be created successfully");

    turboquant::ConversionConfig config;
    config.input_path  = input_dir;
    config.output_path = output_dir;
    config.quantizer.primary_bits  = 4;
    config.quantizer.residual_bits = 4;
    config.quantizer.block_size    = 16;
    config.quantizer.sensitive_layers_start = 1;
    config.quantizer.sensitive_layers_end   = 1;

    bool converted = turboquant::convert_model(config);
    assert(converted && "convert_model must succeed with sensitive layer settings");

    // Load the TQ shard to verify interior layers were quantized
    const std::string tq_shard = output_dir + "/model.safetensors";
    assert(fs::exists(tq_shard) && "TQ output shard must exist");
    auto [tq_tensors, tq_meta] = mlx::core::load_safetensors(tq_shard);

    // Layers 1 and 2 should be TQ-compressed: look for .packed_primary keys
    assert(tq_tensors.count("model.layers.1.self_attn.q_proj.packed_primary") &&
           "layer 1 must be TQ-compressed (packed_primary present)");
    assert(tq_tensors.count("model.layers.2.self_attn.q_proj.packed_primary") &&
           "layer 2 must be TQ-compressed (packed_primary present)");

    // Layers 0 and 3 must NOT appear in the TQ shard as packed weights
    assert(!tq_tensors.count("model.layers.0.self_attn.q_proj.packed_primary") &&
           "layer 0 must NOT be TQ-compressed (sensitive start layer)");
    assert(!tq_tensors.count("model.layers.3.self_attn.q_proj.packed_primary") &&
           "layer 3 must NOT be TQ-compressed (sensitive end layer)");

    // Load the passthrough shard to verify sensitive layers are preserved at fp16
    const std::string pt_shard = output_dir + "/model_passthrough.safetensors";
    assert(fs::exists(pt_shard) && "passthrough shard must exist");
    auto [pt_tensors, pt_meta] = mlx::core::load_safetensors(pt_shard);

    // Layers 0 and 3 should appear in passthrough with their original .weight key
    assert(pt_tensors.count("model.layers.0.self_attn.q_proj.weight") &&
           "layer 0 must be preserved at fp16 in passthrough");
    assert(pt_tensors.count("model.layers.3.self_attn.q_proj.weight") &&
           "layer 3 must be preserved at fp16 in passthrough");

    // Layers 1 and 2 must NOT appear in passthrough
    assert(!pt_tensors.count("model.layers.1.self_attn.q_proj.weight") &&
           "layer 1 must not be in passthrough (should be quantized)");
    assert(!pt_tensors.count("model.layers.2.self_attn.q_proj.weight") &&
           "layer 2 must not be in passthrough (should be quantized)");

    // The 1D layernorm weight must always be in passthrough regardless of
    // sensitive layer settings, because it is not a 2D quantizable weight
    assert(pt_tensors.count("model.layers.0.input_layernorm.weight") &&
           "1D layernorm weight must always be in passthrough");

    fs::remove_all(input_dir);
    fs::remove_all(output_dir);

    printf("  PASS: sensitive layer mixed precision routing\n");
}

int main() {
    printf("test_converter_e2e (integration):\n");
    test_tiny_model_convert_load_forward();
    test_output_tensor_names_and_shapes();
    test_metadata_format();
    test_non_power_of_2_conversion();
    test_sensitive_layer_mixed_precision();
    printf("All converter e2e integration tests passed.\n");
    return 0;
}
