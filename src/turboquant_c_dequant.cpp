/// C API implementation for model dequantization.
/// Reconstructs fp16 weights from TQ-compressed safetensors so downstream
/// loaders (SwiftLM, HuggingFace) can consume the model without TQ support.

#include "turboquant_c/turboquant_c.h"
#include "turboquant/turboquant.h"
#include "turboquant/dequantizer.h"

#include <mlx/mlx.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace turboquant;

extern "C" {

int tq_model_dequant(const char* tq_model_path, const char* output_path) {
    try {
        if (!tq_model_path || !output_path) return 1;
        if (!fs::exists(tq_model_path) || !fs::is_directory(tq_model_path)) return 1;

        fs::create_directories(output_path);

        // Copy non-safetensors files (config.json, tokenizer, generation_config)
        for (const auto& entry : fs::directory_iterator(tq_model_path)) {
            if (entry.is_regular_file() && entry.path().extension() != ".safetensors") {
                fs::copy_file(entry.path(),
                    fs::path(output_path) / entry.path().filename(),
                    fs::copy_options::overwrite_existing);
            }
        }

        // Strip quantization_config from the copied config.json so downstream
        // loaders treat the output as a standard fp16 model
        {
            fs::path config_path = fs::path(output_path) / "config.json";
            if (fs::exists(config_path)) {
                std::ifstream in(config_path);
                std::string json_str((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
                in.close();

                auto qc_start = json_str.find("\"quantization_config\"");
                if (qc_start != std::string::npos) {
                    auto brace_open = json_str.find('{', qc_start);
                    if (brace_open != std::string::npos) {
                        int depth = 1;
                        auto pos = brace_open + 1;
                        while (pos < json_str.size() && depth > 0) {
                            if (json_str[pos] == '{') depth++;
                            else if (json_str[pos] == '}') depth--;
                            pos++;
                        }
                        auto comma_pos = json_str.rfind(',', qc_start);
                        if (comma_pos != std::string::npos && comma_pos < qc_start) {
                            json_str.erase(comma_pos, pos - comma_pos);
                        }
                        std::ofstream out(config_path);
                        out << json_str;
                    }
                }
            }
        }

        // Dequantize each TQ safetensors shard to fp16
        for (const auto& entry : fs::directory_iterator(tq_model_path)) {
            if (entry.path().extension() != ".safetensors") continue;
            std::string fname = entry.path().filename().string();
            if (fname.find("_passthrough") != std::string::npos) continue;

            auto [tensors, metadata] = mlx::core::load_safetensors(entry.path().string());

            if (metadata.count("quantization_method") == 0 ||
                metadata["quantization_method"] != "turboquant") {
                fs::copy_file(entry.path(),
                    fs::path(output_path) / fname,
                    fs::copy_options::overwrite_existing);
                continue;
            }

            auto cb_p_arr = tensors.at("tq_codebook_primary");
            auto cb_r_arr = tensors.at("tq_codebook_residual");
            mlx::core::eval(cb_p_arr);
            mlx::core::eval(cb_r_arr);

            Codebook shared_primary_cb, shared_residual_cb;
            shared_primary_cb.bits = 4;
            shared_residual_cb.bits = 4;
            {
                auto d = cb_p_arr.data<float>();
                for (int i = 0; i < static_cast<int>(cb_p_arr.size()); i++)
                    shared_primary_cb.centroids.push_back(d[i]);
            }
            {
                auto d = cb_r_arr.data<float>();
                for (int i = 0; i < static_cast<int>(cb_r_arr.size()); i++)
                    shared_residual_cb.centroids.push_back(d[i]);
            }

            bool has_per_layer = (metadata.count("tq_per_layer_codebooks") &&
                                  metadata["tq_per_layer_codebooks"] == "true");

            std::unordered_map<std::string, mlx::core::array> output;

            for (const auto& [name, tensor] : tensors) {
                std::string suffix = ".packed_primary";
                if (name.size() < suffix.size() ||
                    name.substr(name.size() - suffix.size()) != suffix) continue;

                std::string layer = name.substr(0, name.size() - suffix.size());

                auto pp = tensors.at(layer + ".packed_primary");
                auto pr = tensors.at(layer + ".packed_residual");
                auto nm = tensors.at(layer + ".norms");
                auto sd = tensors.at(layer + ".seeds");
                mlx::core::eval(pp);
                mlx::core::eval(pr);
                mlx::core::eval(nm);
                mlx::core::eval(sd);

                auto sp = sd.data<uint32_t>();
                uint32_t bs = (sd.size() > 2) ? sp[2] : 512;

                Codebook primary_cb = shared_primary_cb;
                Codebook residual_cb = shared_residual_cb;

                if (has_per_layer) {
                    std::string layer_cb_p_key = layer + ".codebook_primary";
                    std::string layer_cb_r_key = layer + ".codebook_residual";
                    if (tensors.count(layer_cb_p_key)) {
                        auto layer_cb_p = tensors.at(layer_cb_p_key);
                        mlx::core::eval(layer_cb_p);
                        primary_cb.centroids.clear();
                        auto d = layer_cb_p.data<float>();
                        for (int i = 0; i < static_cast<int>(layer_cb_p.size()); i++)
                            primary_cb.centroids.push_back(d[i]);
                    }
                    if (tensors.count(layer_cb_r_key)) {
                        auto layer_cb_r = tensors.at(layer_cb_r_key);
                        mlx::core::eval(layer_cb_r);
                        residual_cb.centroids.clear();
                        auto d = layer_cb_r.data<float>();
                        for (int i = 0; i < static_cast<int>(layer_cb_r.size()); i++)
                            residual_cb.centroids.push_back(d[i]);
                    }
                }

                QuantizedWeight qw{pp, pr, nm, sd};
                auto weight = dequantize_weight_cpu(qw, primary_cb, residual_cb, bs);

                // Preserve original model dtype (typically bfloat16 for HF models).
                // Detect from passthrough tensors which retain the source dtype.
                auto target_dtype = mlx::core::bfloat16;
                std::string pt_check_name = entry.path().stem().string() + "_passthrough.safetensors";
                auto pt_check_path = fs::path(tq_model_path) / pt_check_name;
                if (fs::exists(pt_check_path)) {
                    auto [pt_check, _unused] = mlx::core::load_safetensors(pt_check_path.string());
                    for (auto& [_n, _t] : pt_check) {
                        target_dtype = _t.dtype();
                        break;
                    }
                }

                weight = mlx::core::astype(weight, target_dtype);
                mlx::core::eval(weight);

                output.insert({layer + ".weight", weight});
            }

            // Merge passthrough tensors (sensitive layers kept at fp16)
            std::string pt_name = entry.path().stem().string() + "_passthrough.safetensors";
            auto pt_path = fs::path(tq_model_path) / pt_name;
            if (fs::exists(pt_path)) {
                auto [pt_tensors, _] = mlx::core::load_safetensors(pt_path.string());
                for (auto& [n, t] : pt_tensors) {
                    output.insert({n, t});
                }
            }

            auto out_file = fs::path(output_path) / fname;
            mlx::core::save_safetensors(out_file.string(), output);
        }

        return 0;
    } catch (...) {
        return 1;
    }
}

} // extern "C"
