#pragma once

#include <mlx/mlx.h>
#include <vector>

namespace turboquant {

/// FP16 decode window that shadows the most recent tokens from the
/// TQ compressed KV cache. Incrementally updated during decode to
/// avoid full-cache dequantization on every token.
class DecodeBuffer {
public:
    /// Allocate buffer for the given window size and head configuration.
    DecodeBuffer(int window_size, int num_heads, int head_dim);

    /// Rebuild the entire buffer by dequantizing from compressed cache.
    /// Called once after prefill completes.
    void rebuild(const mlx::core::array& decompressed_keys,
                 const mlx::core::array& decompressed_values);

    /// Append a single token's K/V to the buffer (decode phase).
    /// Only dequantizes the new token, not the full cache.
    void append_token(const mlx::core::array& key, const mlx::core::array& value);

    /// Slide the window forward when the buffer is full.
    /// Dequantizes the next chunk from compressed cache.
    void slide_window(const mlx::core::array& new_keys, const mlx::core::array& new_values);

    /// Get key/value slices for SDPA.
    mlx::core::array keys(int start, int end) const;
    mlx::core::array values(int start, int end) const;

    int window_size() const;
    int current_length() const;

private:
    int window_size_;
    int num_heads_;
    int head_dim_;
    int current_length_ = 0;
    int window_start_ = 0;

    // Flat circular storage: [window_size * num_heads * head_dim] per tensor.
    // Slot index = absolute_token_index % window_size.
    std::vector<float> key_buffer_;
    std::vector<float> value_buffer_;
};

} // namespace turboquant
