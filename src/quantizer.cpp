#include "turboquant/quantizer.h"
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace turboquant {

/// Derive a deterministic seed from the weight matrix contents.
/// This ensures that quantizing the same weight matrix always produces
/// identical results, while different matrices get different rotations.
static uint32_t derive_seed(const mlx::core::array& weight, uint32_t salt) {
    mlx::core::eval(const_cast<mlx::core::array&>(weight));
    const float* data = weight.data<float>();
    size_t n = static_cast<size_t>(weight.size());

    // FNV-1a hash over the raw float bytes for fast deterministic seeding
    uint32_t hash = 2166136261u ^ salt;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t byte_count = n * sizeof(float);
    // Sample at most 4096 bytes to keep overhead bounded on large matrices
    size_t stride = (byte_count > 4096) ? (byte_count / 4096) : 1;
    for (size_t i = 0; i < byte_count; i += stride) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

/// Pack a flat uint8 index array (values 0-15) into nibble pairs.
/// Two consecutive 4-bit indices are stored per byte: lo | (hi << 4).
/// Input length must be even. Output length is input_length / 2.
static std::vector<uint8_t> pack_4bit(const uint8_t* indices, size_t count) {
    std::vector<uint8_t> packed(count / 2);
    for (size_t i = 0; i < count; i += 2) {
        packed[i / 2] = (indices[i] & 0x0F) | ((indices[i + 1] & 0x0F) << 4);
    }
    return packed;
}

QuantizedWeight quantize_weight(
    const mlx::core::array& weight,
    const Codebook& primary_codebook,
    const Codebook& residual_codebook,
    const QuantizerConfig& config) {

    mlx::core::eval(const_cast<mlx::core::array&>(weight));

    const int out_features = static_cast<int>(weight.shape(0));
    const int in_features = static_cast<int>(weight.shape(1));
    const float scale_factor = std::sqrt(static_cast<float>(in_features));
    const float inv_scale = 1.0f / scale_factor;

    // --- Step 1: Extract row norms and normalize ---
    // Compute L2 norm per row: norms[i] = sqrt(sum(w[i,:]^2))
    auto sq = mlx::core::multiply(weight, weight);
    auto row_sum = mlx::core::sum(sq, {1}, /* keepdims= */ true);
    auto orig_norms_2d = mlx::core::sqrt(row_sum); // [out, 1]
    mlx::core::eval(orig_norms_2d);

    // Guard against zero-norm rows to prevent division by zero
    auto safe_norms = mlx::core::maximum(
        orig_norms_2d,
        mlx::core::array(1e-10f));
    auto w_norm = mlx::core::divide(weight, safe_norms); // [out, in]
    mlx::core::eval(w_norm);

    // --- Step 2: Generate deterministic seeds from weight content ---
    uint32_t seed_primary = derive_seed(weight, 0u);
    uint32_t seed_residual = derive_seed(weight, 1u);

    // --- Step 3: Primary pass ---
    // Rotate the normalized weight with Walsh-Hadamard transform
    auto rotated = apply_wht_rotation(w_norm, seed_primary, config.block_size);
    mlx::core::eval(rotated);

    // Scale so coordinates are approximately N(0,1) for the codebook
    auto scaled = mlx::core::multiply(rotated, mlx::core::array(scale_factor));
    mlx::core::eval(scaled);

    // Quantize each element to its nearest codebook centroid
    auto flat_scaled = mlx::core::reshape(scaled, {-1});
    auto indices_primary = turboquant::quantize(flat_scaled, primary_codebook);
    mlx::core::eval(indices_primary);

    // Pack primary indices based on bit width.
    // For bits <= 4, two indices fit in one byte (nibble packing).
    // For bits == 5, each index occupies a full byte (32 centroids need 5 bits,
    // which exceeds a 4-bit nibble, so 3 bits per byte are unused).
    const uint8_t* primary_idx_ptr = indices_primary.data<uint8_t>();
    size_t total_elements = static_cast<size_t>(out_features) * static_cast<size_t>(in_features);

    auto packed_primary = [&]() -> mlx::core::array {
        if (config.primary_bits <= 4) {
            auto packed_primary_vec = pack_4bit(primary_idx_ptr, total_elements);
            return mlx::core::array(
                packed_primary_vec.data(),
                {out_features, in_features / 2},
                mlx::core::uint8);
        } else {
            // 5-bit mode: store one index per byte (raw uint8, no packing)
            return mlx::core::array(
                primary_idx_ptr,
                {out_features, in_features},
                mlx::core::uint8);
        }
    }();

    // --- Step 4: Residual pass (when residual_bits > 0) ---
    // Initialize with zero-filled placeholder; overwritten if residual is enabled
    auto packed_residual = mlx::core::zeros({out_features, in_features / 2}, mlx::core::uint8);
    if (config.residual_bits > 0) {
        // Dequantize primary in the scaled/rotated domain to compute residual
        auto dq_primary = turboquant::dequantize(indices_primary, primary_codebook);
        mlx::core::eval(dq_primary);

        // Compute residual in the scaled, rotated domain (primary WHT space)
        auto residual_flat = mlx::core::subtract(flat_scaled, dq_primary);
        mlx::core::eval(residual_flat);

        auto flat_residual_for_quant = [&]() -> mlx::core::array {
            if (config.shared_rotation) {
                // Shared-rotation mode: the residual is already in the primary's
                // rotated domain. Its distribution is truncated-uniform within each
                // codebook bin (excess kurtosis ~16, not Gaussian). For best quality,
                // the caller should supply a residual codebook fitted to the actual
                // residual distribution via generate_codebook_from_data().
                // No inverse WHT or re-rotation needed — quantize in place for a
                // single-WHT dequant path.
                return residual_flat;
            } else {
                // Legacy dual-rotation mode: transform residual back to the original
                // domain, then re-rotate with a separate seed for an independent WHT
                auto residual_scaled = mlx::core::reshape(residual_flat, {out_features, in_features});
                mlx::core::eval(residual_scaled);
                auto residual_unscaled = mlx::core::multiply(residual_scaled, mlx::core::array(inv_scale));
                mlx::core::eval(residual_unscaled);
                auto residual_orig = apply_inverse_wht_rotation(residual_unscaled, seed_primary, config.block_size);
                mlx::core::eval(residual_orig);
                auto residual_rotated = apply_wht_rotation(residual_orig, seed_residual, config.block_size);
                mlx::core::eval(residual_rotated);
                auto residual_rescaled = mlx::core::multiply(residual_rotated, mlx::core::array(scale_factor));
                mlx::core::eval(residual_rescaled);
                return mlx::core::reshape(residual_rescaled, {-1});
            }
        }();

        auto indices_residual = turboquant::quantize(flat_residual_for_quant, residual_codebook);
        mlx::core::eval(indices_residual);

        const uint8_t* res_idx_ptr = indices_residual.data<uint8_t>();
        auto packed_res_vec = pack_4bit(res_idx_ptr, total_elements);
        packed_residual = mlx::core::array(
            packed_res_vec.data(),
            {out_features, in_features / 2},
            mlx::core::uint8);
    } else {
        // No residual pass: emit a zero-filled placeholder with correct shape
        packed_residual = mlx::core::zeros({out_features, in_features / 2}, mlx::core::uint8);
    }

    // --- Step 5: Norm correction ---
    // Reconstruct the normalized weight from the full quantization (primary +
    // residual when present) to measure how much quantization shrinks each row.
    // The corrected norm compensates for this shrinkage during dequantization.
    auto dq_primary_full = turboquant::dequantize(indices_primary, primary_codebook);
    mlx::core::eval(dq_primary_full);
    auto dq_reshaped = mlx::core::reshape(dq_primary_full, {out_features, in_features});

    // When dual-pass is active, include the residual reconstruction so that
    // the norm correction reflects the actual dequantized output. Without this,
    // the correction only accounts for the primary pass, leaving residual
    // contributions uncompensated and norms less accurate.
    auto recon_normalized = [&]() -> mlx::core::array {
        if (config.residual_bits > 0 && config.shared_rotation) {
            // Shared-rotation mode: primary and residual centroids are both in the
            // primary's rotated domain, so sum them before a single inverse WHT.
            const uint8_t* res_packed_ptr = packed_residual.data<uint8_t>();
            size_t res_packed_count = static_cast<size_t>(out_features) * static_cast<size_t>(in_features / 2);
            std::vector<uint8_t> res_indices_vec(total_elements);
            for (size_t i = 0; i < res_packed_count; ++i) {
                res_indices_vec[i * 2]     = res_packed_ptr[i] & 0x0F;
                res_indices_vec[i * 2 + 1] = (res_packed_ptr[i] >> 4) & 0x0F;
            }
            auto res_indices_arr = mlx::core::array(
                res_indices_vec.data(),
                {static_cast<int>(total_elements)},
                mlx::core::uint8);
            auto dq_residual_full = turboquant::dequantize(res_indices_arr, residual_codebook);
            mlx::core::eval(dq_residual_full);
            auto dq_res_reshaped = mlx::core::reshape(dq_residual_full, {out_features, in_features});

            // Sum in the rotated domain, then unscale and apply single inverse WHT
            auto combined = mlx::core::add(dq_reshaped, dq_res_reshaped);
            mlx::core::eval(combined);
            auto combined_unscaled = mlx::core::multiply(combined, mlx::core::array(inv_scale));
            mlx::core::eval(combined_unscaled);
            auto result = apply_inverse_wht_rotation(combined_unscaled, seed_primary, config.block_size);
            mlx::core::eval(result);
            return result;
        }

        // Primary-only or legacy dual-rotation: inverse WHT primary independently
        auto unscaled_primary = mlx::core::multiply(dq_reshaped, mlx::core::array(inv_scale));
        mlx::core::eval(unscaled_primary);
        auto result = apply_inverse_wht_rotation(unscaled_primary, seed_primary, config.block_size);
        mlx::core::eval(result);

        if (config.residual_bits > 0) {
            // Legacy dual-rotation: residual uses its own seed and inverse WHT
            const uint8_t* res_packed_ptr = packed_residual.data<uint8_t>();
            size_t res_packed_count = static_cast<size_t>(out_features) * static_cast<size_t>(in_features / 2);
            std::vector<uint8_t> res_indices_vec(total_elements);
            for (size_t i = 0; i < res_packed_count; ++i) {
                res_indices_vec[i * 2]     = res_packed_ptr[i] & 0x0F;
                res_indices_vec[i * 2 + 1] = (res_packed_ptr[i] >> 4) & 0x0F;
            }
            auto res_indices_arr = mlx::core::array(
                res_indices_vec.data(),
                {static_cast<int>(total_elements)},
                mlx::core::uint8);
            auto dq_residual_full = turboquant::dequantize(res_indices_arr, residual_codebook);
            mlx::core::eval(dq_residual_full);
            auto dq_res_reshaped = mlx::core::reshape(dq_residual_full, {out_features, in_features});
            auto unscaled_residual = mlx::core::multiply(dq_res_reshaped, mlx::core::array(inv_scale));
            mlx::core::eval(unscaled_residual);
            auto recon_residual = apply_inverse_wht_rotation(unscaled_residual, seed_residual, config.block_size);
            mlx::core::eval(recon_residual);

            result = mlx::core::add(result, recon_residual);
            mlx::core::eval(result);
        }
        return result;
    }();

    // Compute per-row norm of the full reconstruction in normalized space
    auto recon_sq = mlx::core::multiply(recon_normalized, recon_normalized);
    auto recon_row_sum = mlx::core::sum(recon_sq, {1}, /* keepdims= */ true);
    auto recon_norm_2d = mlx::core::sqrt(recon_row_sum); // [out, 1]
    mlx::core::eval(recon_norm_2d);

    // corrected_norm = original_norm / recon_norm, guarding against zero denominator
    auto safe_recon_norm = mlx::core::maximum(
        recon_norm_2d,
        mlx::core::array(1e-10f));
    auto corrected_norms_2d = mlx::core::divide(orig_norms_2d, safe_recon_norm);
    auto corrected_norms = mlx::core::reshape(corrected_norms_2d, {out_features});
    mlx::core::eval(corrected_norms);

    // Store seeds AND the actual block_size used for this layer, so the
    // dequantizer can reconstruct with the correct rotation parameters.
    // When shared_rotation is active, seeds[1] is set to 0 to signal that
    // both passes share the primary WHT domain and only one inverse WHT
    // is needed during dequantization.
    uint32_t stored_residual_seed = (config.shared_rotation && config.residual_bits > 0)
        ? 0u : seed_residual;
    std::vector<uint32_t> seed_vec = {seed_primary, stored_residual_seed, config.block_size};
    auto seeds = mlx::core::array(seed_vec.data(), {3}, mlx::core::uint32);

    return QuantizedWeight{packed_primary, packed_residual, corrected_norms, seeds};
}

} // namespace turboquant
