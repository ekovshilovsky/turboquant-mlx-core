#include "turboquant/dequantizer.h"
#include <mlx/fast.h>
#include <mlx/primitives.h>
#include <mlx/backend/metal/device.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <vector>

namespace turboquant {

/// Metal kernel body for GPU-accelerated codebook lookup.
/// Unpacks 4-bit nibble pairs from packed primary and residual buffers,
/// looks up centroid values in the respective codebooks, and writes the
/// combined (primary + residual) centroid values to the output buffer.
/// The inverse WHT rotation and norm scaling are handled by the caller
/// using MLX operations to ensure numerical consistency with the CPU path.
///
/// MLX wraps this source inside a generated kernel function with buffer
/// parameters derived from the input/output names and thread position
/// attributes. Only the kernel body is provided here.
static const char* kCodebookLookupKernelSource = R"metal(
    uint byte_idx = thread_position_in_grid.x;
    uint out_idx  = byte_idx * 2;

    uint8_t p_byte = packed_primary[byte_idx];
    uint p_lo = p_byte & 0x0F;
    uint p_hi = (p_byte >> 4) & 0x0F;

    uint8_t r_byte = packed_residual[byte_idx];
    uint r_lo = r_byte & 0x0F;
    uint r_hi = (r_byte >> 4) & 0x0F;

    output[out_idx]     = codebook_primary[p_lo] + codebook_residual[r_lo];
    output[out_idx + 1] = codebook_primary[p_hi] + codebook_residual[r_hi];
)metal";

/// Unpack 4-bit nibble pairs into individual uint8 indices.
/// Each input byte yields two output indices: lo nibble then hi nibble.
/// Output length is input_length * 2.
static std::vector<uint8_t> unpack_4bit(const uint8_t* packed, size_t packed_count) {
    std::vector<uint8_t> indices(packed_count * 2);
    for (size_t i = 0; i < packed_count; ++i) {
        indices[i * 2]     = packed[i] & 0x0F;
        indices[i * 2 + 1] = (packed[i] >> 4) & 0x0F;
    }
    return indices;
}

mlx::core::array dequantize_weight_cpu(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size) {

    // Materialize all packed arrays before accessing raw pointers
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    const int out_features = static_cast<int>(qw.packed_primary.shape(0));
    const int packed_cols = static_cast<int>(qw.packed_primary.shape(1));

    // Detect packing mode from packed shape: if packed_cols == in_features,
    // 5-bit mode stores one index per byte. If packed_cols == in_features/2,
    // the standard 4-bit nibble packing is in use.
    // We derive in_features from the residual shape (always nibble-packed)
    // or from the primary shape depending on mode.
    const int residual_packed_cols = static_cast<int>(qw.packed_residual.shape(1));
    const int in_features_from_residual = residual_packed_cols * 2;
    const bool primary_5bit = (packed_cols == in_features_from_residual);
    const int in_features = primary_5bit ? packed_cols : packed_cols * 2;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(in_features));

    // Extract WHT seeds
    const uint32_t* seed_ptr = qw.seeds.data<uint32_t>();
    uint32_t seed_primary = seed_ptr[0];
    uint32_t seed_residual = seed_ptr[1];

    // --- Step 1: Unpack and look up primary centroids ---
    size_t total_elements = static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
    std::vector<float> primary_values(total_elements);
    const auto& p_centroids = primary_codebook.centroids;
    size_t num_centroids = p_centroids.size();

    const uint8_t* primary_packed = qw.packed_primary.data<uint8_t>();
    if (primary_5bit) {
        // 5-bit mode: one index per byte, direct lookup
        for (size_t i = 0; i < total_elements; ++i) {
            size_t idx = primary_packed[i];
            if (idx >= num_centroids) idx = num_centroids - 1;
            primary_values[i] = p_centroids[idx];
        }
    } else {
        // 4-bit mode: unpack nibble pairs then lookup
        size_t packed_total = static_cast<size_t>(out_features) * static_cast<size_t>(packed_cols);
        auto primary_indices = unpack_4bit(primary_packed, packed_total);
        for (size_t i = 0; i < total_elements; ++i) {
            size_t idx = primary_indices[i];
            if (idx >= num_centroids) idx = num_centroids - 1;
            primary_values[i] = p_centroids[idx];
        }
    }

    // Wrap in MLX array for WHT inverse transform
    auto primary_dq = mlx::core::array(
        primary_values.data(),
        {out_features, in_features},
        mlx::core::float32);

    // --- Step 2: Detect residual presence ---
    // Scan the packed residual buffer for any nonzero content. The residual
    // buffer is always nibble-packed (2 indices per byte) regardless of
    // primary packing mode.
    size_t residual_packed_total = static_cast<size_t>(out_features) * static_cast<size_t>(residual_packed_cols);
    bool has_residual = false;
    {
        const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
        for (size_t i = 0; i < residual_packed_total; ++i) {
            if (res_packed[i] != 0) {
                has_residual = true;
                break;
            }
        }
    }

    // Shared-rotation mode is signaled by seed_residual == 0 when residual
    // data is present. Both passes share the primary WHT domain, so we sum
    // centroids before a single inverse WHT instead of running two passes.
    bool shared_rotation = (seed_residual == 0 && has_residual);

    auto result = [&]() -> mlx::core::array {
        if (shared_rotation && !primary_5bit) {
            // --- Shared-rotation path with 256-entry LUT optimization ---
            // Precompute a 256-entry table of primary + residual centroid sums,
            // indexed by (residual_idx << 4 | primary_idx). This replaces per-
            // element codebook lookups and float additions with a single table
            // read, operating directly on packed byte pairs from both buffers.
            const auto& r_centroids = residual_codebook.centroids;

            std::vector<float> lut(256);
            for (int r = 0; r < 16; ++r) {
                for (int p = 0; p < static_cast<int>(p_centroids.size()); ++p) {
                    uint8_t idx = static_cast<uint8_t>((r << 4) | p);
                    lut[idx] = p_centroids[p] + r_centroids[r];
                }
            }

            const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
            std::vector<float> combined_values(total_elements);
            for (int row = 0; row < out_features; ++row) {
                for (int byte_idx = 0; byte_idx < packed_cols; ++byte_idx) {
                    size_t packed_offset = static_cast<size_t>(row) * packed_cols + byte_idx;
                    uint8_t p_byte = primary_packed[packed_offset];
                    uint8_t r_byte = res_packed[packed_offset];

                    // Even column: lo nibbles from both packed bytes
                    size_t col_even = static_cast<size_t>(row) * in_features
                                    + byte_idx * 2;
                    uint8_t combined_even = (p_byte & 0x0F) | ((r_byte & 0x0F) << 4);
                    combined_values[col_even] = lut[combined_even];

                    // Odd column: hi nibbles from both packed bytes
                    size_t col_odd = col_even + 1;
                    uint8_t combined_odd = ((p_byte >> 4) & 0x0F) | (r_byte & 0xF0);
                    combined_values[col_odd] = lut[combined_odd];
                }
            }

            auto combined_dq = mlx::core::array(
                combined_values.data(),
                {out_features, in_features},
                mlx::core::float32);

            // Unscale and apply single inverse WHT on the combined centroids
            auto combined_unscaled = mlx::core::multiply(combined_dq, mlx::core::array(inv_scale));
            mlx::core::eval(combined_unscaled);
            auto r = apply_inverse_wht_rotation(combined_unscaled, seed_primary, block_size);
            mlx::core::eval(r);
            return r;
        }

        if (shared_rotation && primary_5bit) {
            // --- Shared-rotation path for 5-bit primary (no LUT) ---
            // 5-bit primary uses one index per byte, so the nibble-pair LUT
            // optimization does not apply. Fall through to direct centroid lookup.
            const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
            auto residual_indices = unpack_4bit(res_packed, residual_packed_total);

            std::vector<float> combined_values(total_elements);
            const auto& r_centroids = residual_codebook.centroids;
            size_t r_num = r_centroids.size();
            for (size_t i = 0; i < total_elements; ++i) {
                size_t r_idx = residual_indices[i];
                if (r_idx >= r_num) r_idx = r_num - 1;
                combined_values[i] = primary_values[i] + r_centroids[r_idx];
            }

            auto combined_dq = mlx::core::array(
                combined_values.data(),
                {out_features, in_features},
                mlx::core::float32);

            auto combined_unscaled = mlx::core::multiply(combined_dq, mlx::core::array(inv_scale));
            mlx::core::eval(combined_unscaled);
            auto r = apply_inverse_wht_rotation(combined_unscaled, seed_primary, block_size);
            mlx::core::eval(r);
            return r;
        }

        // --- Legacy dual-rotation path: two independent inverse WHTs ---
        auto primary_unscaled = mlx::core::multiply(primary_dq, mlx::core::array(inv_scale));
        mlx::core::eval(primary_unscaled);
        auto r = apply_inverse_wht_rotation(primary_unscaled, seed_primary, block_size);
        mlx::core::eval(r);

        if (has_residual) {
            const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
            auto residual_indices = unpack_4bit(res_packed, residual_packed_total);

            std::vector<float> residual_values(total_elements);
            const auto& r_centroids = residual_codebook.centroids;
            size_t r_num = r_centroids.size();
            for (size_t i = 0; i < total_elements; ++i) {
                size_t idx = residual_indices[i];
                if (idx >= r_num) idx = r_num - 1;
                residual_values[i] = r_centroids[idx];
            }

            auto residual_dq = mlx::core::array(
                residual_values.data(),
                {out_features, in_features},
                mlx::core::float32);

            auto residual_unscaled = mlx::core::multiply(residual_dq, mlx::core::array(inv_scale));
            mlx::core::eval(residual_unscaled);
            auto residual_orig = apply_inverse_wht_rotation(residual_unscaled, seed_residual, block_size);
            mlx::core::eval(residual_orig);

            r = mlx::core::add(r, residual_orig);
            mlx::core::eval(r);
        }
        return r;
    }();

    // --- Step 4: Apply corrected norms ---
    // norms has shape [out], broadcast multiply against [out, in]
    auto norms_2d = mlx::core::reshape(qw.norms, {out_features, 1});
    result = mlx::core::multiply(result, norms_2d);
    mlx::core::eval(result);

    return result;
}

/// Metal kernel body for 5-bit GPU-accelerated codebook lookup.
/// Each thread handles one element (one index per byte), unlike the 4-bit
/// kernel which processes two elements per byte via nibble pairs.
static const char* kCodebookLookup5BitKernelSource = R"metal(
    uint elem_idx = thread_position_in_grid.x;

    uint8_t p_idx = packed_primary[elem_idx];
    output[elem_idx] = codebook_primary[p_idx];
)metal";

/// Dispatch the codebook lookup Metal kernel for a single packed buffer and
/// codebook pair. Returns the dequantized centroid values as a float array
/// of shape [out_features, in_features].
///
/// When primary_5bit is true, the packed buffer stores one index per byte
/// and the kernel processes one element per thread instead of two.
static mlx::core::array dispatch_codebook_lookup(
    const mlx::core::array& packed,
    const mlx::core::array& codebook,
    int out_features,
    int in_features,
    bool primary_5bit = false) {

    auto gpu_stream = mlx::core::default_stream(mlx::core::Device::gpu);

    if (primary_5bit) {
        // 5-bit mode: one index per byte, one element per thread
        size_t total_elements = static_cast<size_t>(out_features) * static_cast<size_t>(in_features);
        int grid_threads = static_cast<int>(total_elements);
        int threadgroup_size = std::min(256, grid_threads);

        static auto lookup_kernel_5bit = mlx::core::fast::metal_kernel(
            "tq_codebook_lookup_5bit",
            {"packed_primary", "codebook_primary"},
            {"output"},
            kCodebookLookup5BitKernelSource,
            "",
            true);

        auto results = lookup_kernel_5bit(
            {packed, codebook},
            {{out_features, in_features}},
            {mlx::core::float32},
            std::make_tuple(grid_threads, 1, 1),
            std::make_tuple(threadgroup_size, 1, 1),
            {},
            std::nullopt,
            false,
            gpu_stream);

        return results[0];
    }

    // 4-bit mode: two indices per byte (nibble pairs)
    int packed_cols = in_features / 2;
    size_t total_packed = static_cast<size_t>(out_features) * static_cast<size_t>(packed_cols);
    int grid_threads = static_cast<int>(total_packed);
    int threadgroup_size = std::min(256, grid_threads);

    // Zero codebook used as a placeholder for the unused residual slot.
    // The kernel signature requires both primary and residual codebooks, so
    // we supply a zeroed array when only one pass is needed.
    auto zero_packed = mlx::core::zeros_like(packed);
    auto zero_codebook = mlx::core::zeros_like(codebook);

    static auto lookup_kernel = mlx::core::fast::metal_kernel(
        "tq_codebook_lookup",
        {"packed_primary", "packed_residual", "codebook_primary", "codebook_residual"},
        {"output"},
        kCodebookLookupKernelSource,
        "",
        true);

    // The grid parameter specifies the total number of threads to launch.
    // Each thread processes one packed byte (two weight elements).
    auto results = lookup_kernel(
        {packed, zero_packed, codebook, zero_codebook},
        {{out_features, in_features}},
        {mlx::core::float32},
        std::make_tuple(grid_threads, 1, 1),
        std::make_tuple(threadgroup_size, 1, 1),
        {},
        std::nullopt,
        false,
        gpu_stream);

    return results[0];
}

mlx::core::array dequantize_weight_gpu(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size) {

    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    const int out_features = static_cast<int>(qw.packed_primary.shape(0));
    const int packed_cols = static_cast<int>(qw.packed_primary.shape(1));

    // Detect packing mode from shapes: compare primary packed columns to
    // residual packed columns (residual is always nibble-packed as 2 per byte)
    const int residual_packed_cols = static_cast<int>(qw.packed_residual.shape(1));
    const int in_features_from_residual = residual_packed_cols * 2;
    const bool primary_5bit = (packed_cols == in_features_from_residual);
    const int in_features = primary_5bit ? packed_cols : packed_cols * 2;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(in_features));

    const uint32_t* seed_ptr = qw.seeds.data<uint32_t>();
    uint32_t seed_primary = seed_ptr[0];
    uint32_t seed_residual = seed_ptr[1];

    // Build codebook arrays for the Metal kernel to index into.
    auto cb_primary = mlx::core::array(
        primary_codebook.centroids.data(),
        {static_cast<int>(primary_codebook.centroids.size())},
        mlx::core::float32);
    auto cb_residual = mlx::core::array(
        residual_codebook.centroids.data(),
        {static_cast<int>(residual_codebook.centroids.size())},
        mlx::core::float32);

    // --- Step 1: Primary codebook lookup via Metal kernel ---
    auto primary_dq = dispatch_codebook_lookup(
        qw.packed_primary, cb_primary, out_features, in_features, primary_5bit);

    // --- Step 2: Add residual contribution if present ---
    // Detect whether the residual pass produced quantized values by scanning
    // the packed residual buffer for any nonzero byte. The residual buffer
    // is always nibble-packed regardless of primary packing mode.
    size_t residual_total_packed = static_cast<size_t>(out_features) * static_cast<size_t>(residual_packed_cols);
    bool has_residual = false;
    {
        const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
        for (size_t i = 0; i < residual_total_packed; ++i) {
            if (res_packed[i] != 0) {
                has_residual = true;
                break;
            }
        }
    }

    // Shared-rotation mode is signaled by seed_residual == 0 when residual
    // data is present. Both passes share the primary WHT domain, so we sum
    // centroids before a single inverse WHT instead of running two passes.
    bool shared_rotation = (seed_residual == 0 && has_residual);

    auto result = [&]() -> mlx::core::array {
        if (shared_rotation) {
            // Shared-rotation path: dispatch residual codebook lookup via Metal,
            // sum with primary centroids, then apply a single inverse WHT.
            auto residual_dq = dispatch_codebook_lookup(
                qw.packed_residual, cb_residual, out_features, in_features);

            auto combined = mlx::core::add(primary_dq, residual_dq);
            mlx::core::eval(combined);
            auto combined_unscaled = mlx::core::multiply(combined, mlx::core::array(inv_scale));
            mlx::core::eval(combined_unscaled);
            auto r = apply_inverse_wht_rotation(combined_unscaled, seed_primary, block_size);
            mlx::core::eval(r);
            return r;
        }

        // Legacy dual-rotation path: each pass gets its own inverse WHT
        auto primary_unscaled = mlx::core::multiply(primary_dq, mlx::core::array(inv_scale));
        mlx::core::eval(primary_unscaled);
        auto r = apply_inverse_wht_rotation(primary_unscaled, seed_primary, block_size);
        mlx::core::eval(r);

        if (has_residual) {
            auto residual_dq = dispatch_codebook_lookup(
                qw.packed_residual, cb_residual, out_features, in_features);

            auto residual_unscaled = mlx::core::multiply(residual_dq, mlx::core::array(inv_scale));
            mlx::core::eval(residual_unscaled);
            auto residual_orig = apply_inverse_wht_rotation(residual_unscaled, seed_residual, block_size);
            mlx::core::eval(residual_orig);

            r = mlx::core::add(r, residual_orig);
            mlx::core::eval(r);
        }
        return r;
    }();

    // --- Step 3: Apply corrected norms ---
    auto norms_2d = mlx::core::reshape(qw.norms, {out_features, 1});
    result = mlx::core::multiply(result, norms_2d);
    mlx::core::eval(result);

    return result;
}

// ---------------------------------------------------------------------------
// FusedDequantMatmul — Primitive-based Metal dispatch from compiled metallib
// ---------------------------------------------------------------------------
// Dispatches the tq_dequant_mm kernel from the pre-compiled
// turboquant_mlx.metallib instead of JIT-compiling an inline source string
// via mx.fast.metal_kernel. Benefits:
//   - Kernel is compiled once by CMake with full Metal compiler optimizations
//   - No per-session JIT compilation overhead
//   - Access to Metal compiler features unavailable through JIT (link-time
//     optimization, specialized function constants, etc.)
//
// The dispatch follows MLX's standard Primitive pattern:
//   1. Subclass mlx::core::Primitive with eval_gpu()
//   2. Load the metallib via metal::Device::get_library()
//   3. Obtain the pipeline state via metal::Device::get_kernel()
//   4. Bind buffers and dispatch via metal::CommandEncoder

/// MLX Primitive that dispatches the fused dequant-matmul kernel from the
/// pre-compiled turboquant_mlx.metallib. Each eval_gpu() call binds the
/// compressed weight buffers, codebooks, parameters, and activation input
/// to the tq_dequant_mm kernel function and dispatches one threadgroup per
/// (output_row, batch) pair.
class FusedDequantMatmulPrimitive : public mlx::core::Primitive {
public:
    FusedDequantMatmulPrimitive(
        mlx::core::Stream stream,
        int out_features,
        int in_features,
        int batch,
        uint32_t block_size,
        std::string metallib_path)
        : mlx::core::Primitive(stream)
        , out_features_(out_features)
        , in_features_(in_features)
        , batch_(batch)
        , block_size_(block_size)
        , metallib_path_(std::move(metallib_path)) {}

    void eval_cpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override {
        throw std::runtime_error(
            "FusedDequantMatmul requires GPU — CPU fallback not implemented");
    }

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override {
        // Input layout (must match buffer indices in tq_dequant_mm.metal):
        //   [0] packed_primary    uint8   [out_features, packed_cols]
        //   [1] packed_residual   uint8   [out_features, in_features/2]
        //   [2] codebook_primary  float32 [16] or [32]
        //   [3] codebook_residual float32 [16]
        //   [4] norms             float32 [out_features]
        //   [5] params            uint32  [7]
        //   [6] x                 float16 [batch, in_features]
        //   [7] lut               float32 [256] (precomputed centroid sums for
        //                                  shared-rotation 4-bit LUT path;
        //                                  scalar placeholder otherwise)
        // Output:
        //   [0] y                 float16 [batch, out_features]

        auto& y = outputs[0];
        y.set_data(mlx::core::allocator::malloc(y.nbytes()));

        auto& d = mlx::core::metal::device(mlx::core::Device::gpu);

        // Load the pre-compiled metallib. MLX caches the library after the
        // first load, so subsequent calls return the cached MTL::Library*.
        // When metallib_path_ is a full .metallib path, MLX loads it directly.
        // When empty, MLX searches colocated with the current binary.
        auto* lib = d.get_library("turboquant_mlx", metallib_path_);
        auto* kernel = d.get_kernel("tq_dequant_mm", lib);

        auto& enc = d.get_command_encoder(stream().index);
        enc.set_compute_pipeline_state(kernel);

        // Bind input arrays to buffer slots matching the .metal declaration
        enc.set_input_array(inputs[0], 0);  // packed_primary
        enc.set_input_array(inputs[1], 1);  // packed_residual
        enc.set_input_array(inputs[2], 2);  // codebook_primary
        enc.set_input_array(inputs[3], 3);  // codebook_residual
        enc.set_input_array(inputs[4], 4);  // norms
        enc.set_input_array(inputs[5], 5);  // params
        enc.set_input_array(inputs[6], 6);  // x (activation input)

        enc.set_output_array(y, 7);         // y (output)

        enc.set_input_array(inputs[7], 8);  // lut (256-entry centroid sum table)

        // Dispatch one threadgroup per (output_row, batch) pair.
        // dispatchThreadgroups semantics: grid_dims is the threadgroup count.
        int tg_width = static_cast<int>(
            std::min(block_size_, static_cast<uint32_t>(512)));
        MTL::Size grid_dims = MTL::Size(out_features_, batch_, 1);
        MTL::Size group_dims = MTL::Size(tg_width, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    const char* name() const override { return "FusedDequantMatmul"; }
    bool is_equivalent(const mlx::core::Primitive& other) const override {
        auto* o = dynamic_cast<const FusedDequantMatmulPrimitive*>(&other);
        return o && o->out_features_ == out_features_
            && o->in_features_ == in_features_
            && o->batch_ == batch_
            && o->block_size_ == block_size_;
    }

private:
    int out_features_;
    int in_features_;
    int batch_;
    uint32_t block_size_;
    std::string metallib_path_;
};

/// Resolve the absolute path to turboquant_mlx.metallib. MLX's default
/// colocated search resolves relative to the MLX dylib, not the calling
/// library, so we must resolve the path ourselves.
///
/// Search order:
///   1. TQ_METALLIB_DIR environment variable (build system or test override)
///   2. Directory containing libturboquant_mlx.dylib (resolved via dladdr)
///   3. Directory containing the current executable
///
/// Returns the full path to the .metallib file. MLX's get_library()
/// accepts a path ending in .metallib and loads it directly.
static std::string find_metallib_path() {
    const char* env_dir = std::getenv("TQ_METALLIB_DIR");
    if (env_dir && env_dir[0] != '\0') {
        auto p = std::filesystem::path(env_dir) / "turboquant_mlx.metallib";
        if (std::filesystem::exists(p)) {
            return p.string();
        }
    }

    // Resolve the directory containing libturboquant_mlx.dylib using dladdr
    // on a symbol known to live in this translation unit.
    Dl_info info;
    if (dladdr(reinterpret_cast<const void*>(&find_metallib_path), &info)
        && info.dli_fname) {
        auto lib_dir = std::filesystem::path(info.dli_fname).parent_path();
        auto candidate = lib_dir / "turboquant_mlx.metallib";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    // Fallback: empty string triggers MLX's colocated search (may fail if
    // the metallib is not next to the MLX library itself).
    return "";
}

mlx::core::array fused_dequant_matmul(
    const QuantizedWeight& qw,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    uint32_t block_size,
    const mlx::core::array& input,
    uint32_t full_in_features) {

    // Materialize all compressed weight buffers before reading raw pointers
    mlx::core::eval(qw.packed_primary);
    mlx::core::eval(qw.packed_residual);
    mlx::core::eval(qw.norms);
    mlx::core::eval(qw.seeds);

    const int out_features = static_cast<int>(qw.packed_primary.shape(0));
    const int packed_cols  = static_cast<int>(qw.packed_primary.shape(1));

    // Detect packing mode: compare primary packed columns to residual
    const int residual_packed_cols = static_cast<int>(qw.packed_residual.shape(1));
    const int in_features_from_residual = residual_packed_cols * 2;
    const bool primary_5bit = (packed_cols == in_features_from_residual);
    const int in_features = primary_5bit ? packed_cols : packed_cols * 2;

    const uint32_t* seed_ptr = qw.seeds.data<uint32_t>();
    uint32_t seed_primary = seed_ptr[0];
    uint32_t seed_residual = seed_ptr[1];

    // Detect whether the residual pass produced quantized values
    size_t residual_total_packed = static_cast<size_t>(out_features) * static_cast<size_t>(residual_packed_cols);
    uint32_t has_residual = 0;
    {
        const uint8_t* res_packed = qw.packed_residual.data<uint8_t>();
        for (size_t i = 0; i < residual_total_packed; ++i) {
            if (res_packed[i] != 0) {
                has_residual = 1;
                break;
            }
        }
    }

    auto cb_primary = mlx::core::array(
        primary_codebook.centroids.data(),
        {static_cast<int>(primary_codebook.centroids.size())},
        mlx::core::float32);

    auto cb_residual = mlx::core::array(
        residual_codebook.centroids.data(),
        {static_cast<int>(residual_codebook.centroids.size())},
        mlx::core::float32);

    // Runtime parameters packed into a uint32 array for the kernel to decode.
    // Layout matches the .metal kernel's params buffer expectation.
    //
    // full_in_features defaults to the rank-local in_features whenever the
    // caller did not specify a different original layer size. The kernel's
    // combined_scale needs the unsharded value on row-parallel shards so
    // the per-row norm correction (baked against the whole layer) aligns
    // with the rank-local dequantised weight values.
    uint32_t primary_bits_val = primary_5bit ? 5u : 4u;
    uint32_t full_in_feat_val = (full_in_features == 0u)
        ? static_cast<uint32_t>(in_features)
        : full_in_features;
    std::vector<uint32_t> param_data = {
        block_size,
        static_cast<uint32_t>(out_features),
        static_cast<uint32_t>(in_features),
        has_residual,
        seed_primary,
        seed_residual,
        primary_bits_val,
        full_in_feat_val
    };
    auto params = mlx::core::array(param_data.data(), {8}, mlx::core::uint32);

    // Precompute the 256-entry LUT for shared-rotation 4-bit mode.
    // Each entry holds the sum of one primary centroid and one residual centroid:
    //   lut[(r_idx << 4) | p_idx] = primary_centroid[p_idx] + residual_centroid[r_idx]
    // This eliminates two codebook reads and a float addition per element in the
    // Metal kernel, replacing them with a single threadgroup shared memory read.
    bool shared_rotation = (seed_residual == 0 && has_residual != 0);
    bool use_lut = shared_rotation && !primary_5bit;

    mlx::core::array lut_arr = mlx::core::array({0.0f}, {1}, mlx::core::float32);
    if (use_lut) {
        std::vector<float> lut_data(256);
        for (int r = 0; r < 16; ++r) {
            for (int p = 0; p < 16; ++p) {
                uint8_t idx = static_cast<uint8_t>((r << 4) | p);
                lut_data[idx] = primary_codebook.centroids[p]
                              + residual_codebook.centroids[r];
            }
        }
        lut_arr = mlx::core::array(lut_data.data(), {256}, mlx::core::float32);
    }

    // Cast the input to float16 for the kernel's half-precision activation path
    auto x = mlx::core::astype(input, mlx::core::float16);
    mlx::core::eval(x);

    int batch = static_cast<int>(x.shape(0));

    // Resolve the metallib location once per process lifetime
    static std::string metallib_path = find_metallib_path();

    auto gpu_stream = mlx::core::Stream(
        mlx::core::default_stream(mlx::core::Device::gpu));

    // Schedule the primitive into MLX's compute graph. The output array is
    // allocated lazily by the framework and populated when eval_gpu() runs.
    std::shared_ptr<mlx::core::Primitive> prim =
        std::make_shared<FusedDequantMatmulPrimitive>(
            gpu_stream,
            out_features,
            in_features,
            batch,
            block_size,
            metallib_path);

    auto output = mlx::core::array(
        {batch, out_features},
        mlx::core::float16,
        std::move(prim),
        {qw.packed_primary, qw.packed_residual, cb_primary, cb_residual,
         qw.norms, params, x, lut_arr});

    return output;
}

} // namespace turboquant
