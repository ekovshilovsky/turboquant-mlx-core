#include "turboquant/decode_buffer.h"

#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace turboquant {

DecodeBuffer::DecodeBuffer(int window_size, int num_heads, int head_dim)
    : window_size_(window_size)
    , num_heads_(num_heads)
    , head_dim_(head_dim)
    , key_buffer_(static_cast<size_t>(window_size * num_heads * head_dim), 0.0f)
    , value_buffer_(static_cast<size_t>(window_size * num_heads * head_dim), 0.0f) {}

void DecodeBuffer::rebuild(const mlx::core::array& decompressed_keys,
                           const mlx::core::array& decompressed_values) {
    // Copy decompressed KV data into the circular buffer starting at slot 0.
    // Called once after prefill when the decode window is cold.
    auto keys_f32 = mlx::core::astype(decompressed_keys, mlx::core::float32);
    auto vals_f32 = mlx::core::astype(decompressed_values, mlx::core::float32);
    mlx::core::eval(keys_f32, vals_f32);

    const int num_tokens = static_cast<int>(decompressed_keys.shape(0));
    const int tokens_to_copy = std::min(num_tokens, window_size_);
    const size_t elems_per_token =
        static_cast<size_t>(num_heads_ * head_dim_);

    const float* k_src = keys_f32.data<float>();
    const float* v_src = vals_f32.data<float>();

    for (int t = 0; t < tokens_to_copy; ++t) {
        size_t slot = static_cast<size_t>(t % window_size_);
        size_t buf_offset = slot * elems_per_token;
        size_t src_offset = static_cast<size_t>(t) * elems_per_token;
        std::memcpy(key_buffer_.data() + buf_offset,
                    k_src + src_offset,
                    elems_per_token * sizeof(float));
        std::memcpy(value_buffer_.data() + buf_offset,
                    v_src + src_offset,
                    elems_per_token * sizeof(float));
    }

    current_length_ = num_tokens;
    window_start_ = 0;
}

void DecodeBuffer::append_token(const mlx::core::array& key,
                                const mlx::core::array& value) {
    // Write the new token's K/V into the circular buffer slot for the current
    // position. Only the incoming token is materialised — no full-cache read.
    auto key_f32 = mlx::core::astype(key, mlx::core::float32);
    auto val_f32 = mlx::core::astype(value, mlx::core::float32);
    mlx::core::eval(key_f32, val_f32);

    const size_t elems_per_token =
        static_cast<size_t>(num_heads_ * head_dim_);
    size_t slot = static_cast<size_t>(current_length_ % window_size_);
    size_t buf_offset = slot * elems_per_token;

    std::memcpy(key_buffer_.data() + buf_offset,
                key_f32.data<float>(),
                elems_per_token * sizeof(float));
    std::memcpy(value_buffer_.data() + buf_offset,
                val_f32.data<float>(),
                elems_per_token * sizeof(float));

    ++current_length_;
}

void DecodeBuffer::slide_window(const mlx::core::array& new_keys,
                                const mlx::core::array& new_values) {
    // Overwrite the oldest region of the circular buffer with freshly
    // dequantised data.  This is called when the sliding decode window
    // advances and stale entries must be replaced.
    auto keys_f32 = mlx::core::astype(new_keys, mlx::core::float32);
    auto vals_f32 = mlx::core::astype(new_values, mlx::core::float32);
    mlx::core::eval(keys_f32, vals_f32);

    const int num_tokens = static_cast<int>(new_keys.shape(0));
    const size_t elems_per_token =
        static_cast<size_t>(num_heads_ * head_dim_);

    const float* k_src = keys_f32.data<float>();
    const float* v_src = vals_f32.data<float>();

    // Write starting from window_start_, wrapping as needed
    for (int t = 0; t < num_tokens; ++t) {
        size_t slot = static_cast<size_t>((window_start_ + t) % window_size_);
        size_t buf_offset = slot * elems_per_token;
        size_t src_offset = static_cast<size_t>(t) * elems_per_token;
        std::memcpy(key_buffer_.data() + buf_offset,
                    k_src + src_offset,
                    elems_per_token * sizeof(float));
        std::memcpy(value_buffer_.data() + buf_offset,
                    v_src + src_offset,
                    elems_per_token * sizeof(float));
    }

    window_start_ = (window_start_ + num_tokens) % window_size_;
}

static mlx::core::array extract_range(
    const std::vector<float>& buffer,
    int start, int end,
    int window_size, int num_heads, int head_dim) {

    const int num_tokens = end - start;
    const size_t elems_per_token =
        static_cast<size_t>(num_heads * head_dim);
    std::vector<float> out(
        static_cast<size_t>(num_tokens) * elems_per_token);

    for (int t = 0; t < num_tokens; ++t) {
        // Map the logical absolute token index to a circular buffer slot.
        size_t slot = static_cast<size_t>((start + t) % window_size);
        size_t buf_offset = slot * elems_per_token;
        size_t out_offset = static_cast<size_t>(t) * elems_per_token;
        std::memcpy(out.data() + out_offset,
                    buffer.data() + buf_offset,
                    elems_per_token * sizeof(float));
    }

    auto result = mlx::core::array(
        out.data(),
        {num_tokens, num_heads, head_dim},
        mlx::core::float32);
    return mlx::core::astype(result, mlx::core::float16);
}

mlx::core::array DecodeBuffer::keys(int start, int end) const {
    if (start < 0 || end <= start || end > current_length_) {
        throw std::out_of_range("DecodeBuffer::keys: requested range is out of bounds");
    }
    return extract_range(key_buffer_, start, end, window_size_, num_heads_, head_dim_);
}

mlx::core::array DecodeBuffer::values(int start, int end) const {
    if (start < 0 || end <= start || end > current_length_) {
        throw std::out_of_range("DecodeBuffer::values: requested range is out of bounds");
    }
    return extract_range(value_buffer_, start, end, window_size_, num_heads_, head_dim_);
}

int DecodeBuffer::window_size() const { return window_size_; }
int DecodeBuffer::current_length() const { return current_length_; }

} // namespace turboquant
