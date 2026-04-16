#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "turboquant/turboquant.h"

namespace nb = nanobind;

NB_MODULE(turboquant_mlx, m) {
    m.doc() = "TurboQuant-MLX: Near-lossless weight and KV cache compression for Apple Silicon";

    m.def("version", &turboquant::version, "Library version string");

    // Codebook: Lloyd-Max optimal centroids for quantization
    nb::class_<turboquant::Codebook>(m, "Codebook")
        .def_ro("centroids", &turboquant::Codebook::centroids,
                "Sorted centroid values (2^bits entries)")
        .def_ro("boundaries", &turboquant::Codebook::boundaries,
                "Decision boundaries between adjacent centroids")
        .def_ro("bits", &turboquant::Codebook::bits,
                "Quantization bit width (1-4)");

    m.def("generate_codebook", &turboquant::generate_codebook,
          nb::arg("bits"),
          "Generate Lloyd-Max codebook for the specified bit width");

    m.def("validate_codebook", &turboquant::validate_codebook,
          nb::arg("codebook"),
          "Validate codebook invariants: symmetry, sorting, boundary placement");

    // Model conversion: quantize HuggingFace model to TurboQuant format
    m.def("convert_model", [](const std::string& input_path,
                               const std::string& output_path,
                               int primary_bits,
                               int residual_bits,
                               int block_size) -> bool {
        turboquant::ConversionConfig config;
        config.input_path = input_path;
        config.output_path = output_path;
        config.quantizer.primary_bits = static_cast<uint8_t>(primary_bits);
        config.quantizer.residual_bits = static_cast<uint8_t>(residual_bits);
        config.quantizer.block_size = static_cast<uint32_t>(block_size);
        return turboquant::convert_model(config);
    },
          nb::arg("input_path"),
          nb::arg("output_path"),
          nb::arg("primary_bits") = 4,
          nb::arg("residual_bits") = 4,
          nb::arg("block_size") = 512,
          "Convert a HuggingFace model to TurboQuant format");

    // Model validation: verify converted model metadata and tensor shapes
    m.def("validate_model", &turboquant::validate_converted_model,
          nb::arg("model_path"),
          "Validate that a converted model has correct TQ metadata and tensor shapes");
}
