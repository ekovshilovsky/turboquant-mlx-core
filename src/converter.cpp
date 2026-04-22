#include "turboquant/converter.h"
#include "turboquant/codebook.h"
#include "turboquant/convert/emit_sidecar.h"
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

// Shard strategy inference, safetensors header parsing, and the full sidecar
// emission routine are shared with the tq-emit-sidecar retrofit tool. They
// live in the convert/ subdirectory so both the inline post-write path and
// the retrofit tool produce byte-identical sidecars for the same on-disk
// layout.

/// Emit tq_shard_metadata.json next to the converted safetensors files. This
/// thin wrapper exists to adapt the free-standing retrofit API to the
/// ConversionConfig the inline conversion pass already holds; the real work
/// is in convert::emit_sidecar_for_directory.
static void write_shard_metadata_file(const ConversionConfig& config) {
    (void)convert::emit_sidecar_for_directory(config.output_path);
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
