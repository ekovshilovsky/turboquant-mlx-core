#include "turboquant/quantizer.h"
#include <mlx/fast.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

namespace turboquant {

/// Fast Walsh-Hadamard Transform applied in-place using a butterfly network.
/// Requires n to be a power of two.
static void fwht_inplace(float* data, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                float a = data[j];
                float b = data[j + h];
                data[j]     = a + b;
                data[j + h] = a - b;
            }
        }
    }
}

/// Generate a deterministic sign vector from seed and element index using a
/// hash function identical to the Metal kernel's tq_sign(). Each element is
/// derived independently from (seed, index) with no sequential state, so
/// there is no short-range correlation between adjacent seeds and the CPU
/// and GPU produce bit-identical sign vectors for the same inputs.
static std::vector<float> make_signs(uint32_t seed, uint32_t block_size) {
    std::vector<float> signs(block_size);
    for (uint32_t i = 0; i < block_size; ++i) {
        uint32_t h = seed * 2654435761u + i * 2246822519u;
        h ^= h >> 16;
        h *= 0x45d9f3bu;
        h ^= h >> 16;
        signs[i] = (h & 1u) ? 1.0f : -1.0f;
    }
    return signs;
}

// ---------------------------------------------------------------------------
// GPU WHT rotation via Metal kernel (JIT-compiled through MLX fast API)
// ---------------------------------------------------------------------------
// Dispatches the forward Walsh-Hadamard transform as a Metal compute kernel
// for matrices exceeding a size threshold. Each threadgroup processes one block
// of block_size elements: applies the sign vector, executes log2(block_size)
// butterfly stages in shared memory, and writes the scaled result back.
//
// For matrices below the GPU dispatch threshold (< 1024 total elements), the
// CPU path is used to avoid kernel launch overhead exceeding computation time.

/// Metal kernel source for forward WHT rotation.
/// Each threadgroup processes one block of block_size elements. Thread tid
/// within the group handles element tid of the block: sign multiply, in-place
/// butterfly stages with threadgroup barriers, and final 1/sqrt(n) scaling.
/// The threadgroup shared buffer is declared inside the kernel body because
/// MLX's metal_kernel wraps the body inside a kernel function; program-scope
/// threadgroup declarations are invalid in Metal.
static const char* kWhtForwardKernelSource = R"metal(
    threadgroup float shared_buf[TQ_WHT_BLOCK_SIZE];
    uint block_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint n = block_size_param[0];
    uint seed_val = seed_param[0];

    // Global offset: each threadgroup handles one block of n elements
    uint global_offset = block_idx * n + tid;

    // Load element into threadgroup shared memory with sign flip applied
    uint local_idx = tid;
    uint h = seed_val * 2654435761u + local_idx * 2246822519u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    float sign_val = (h & 1u) ? 1.0f : -1.0f;

    shared_buf[tid] = matrix[global_offset] * sign_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages: log2(n) iterations
    for (uint step = 1; step < n; step <<= 1) {
        if ((tid % (2 * step)) < step) {
            float a = shared_buf[tid];
            float b = shared_buf[tid + step];
            shared_buf[tid] = a + b;
            shared_buf[tid + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by 1/sqrt(n) and write back
    float inv_sqrt_n = rsqrt(float(n));
    output[global_offset] = shared_buf[tid] * inv_sqrt_n;
)metal";

/// Metal kernel source for inverse WHT rotation.
/// The inverse is: FWHT first, then scale by 1/sqrt(n) and multiply by signs.
/// Since WHT is self-inverse up to scaling, the inverse operation is:
/// x_orig = signs * (1/sqrt(n)) * FWHT(x_rotated)
static const char* kWhtInverseKernelSource = R"metal(
    threadgroup float shared_buf[TQ_WHT_BLOCK_SIZE];
    uint block_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint n = block_size_param[0];
    uint seed_val = seed_param[0];

    uint global_offset = block_idx * n + tid;

    // Load element into shared memory (no sign flip yet for inverse)
    shared_buf[tid] = matrix[global_offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages: log2(n) iterations
    for (uint step = 1; step < n; step <<= 1) {
        if ((tid % (2 * step)) < step) {
            float a = shared_buf[tid];
            float b = shared_buf[tid + step];
            shared_buf[tid] = a + b;
            shared_buf[tid + step] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute sign for this element index (within the block, not global)
    uint local_idx = tid;
    uint h = seed_val * 2654435761u + local_idx * 2246822519u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    float sign_val = (h & 1u) ? 1.0f : -1.0f;

    // Scale by 1/sqrt(n) and apply sign
    float inv_sqrt_n = rsqrt(float(n));
    output[global_offset] = shared_buf[tid] * inv_sqrt_n * sign_val;
)metal";

/// Minimum total element count to justify GPU dispatch overhead.
/// Below this threshold the CPU path runs faster than Metal kernel launch.
static constexpr int kGpuDispatchThreshold = 1024;

/// Thread-local flag to force CPU-only execution across all GPU-accelerated
/// operations. The converter's parallel workers set this to prevent GPU Metal
/// dispatch from multiple threads, which would cause command buffer conflicts
/// in MLX's Metal backend. The GPU stream's internal state (command buffers,
/// encoders) is not protected by user-level mutexes, so concurrent GPU dispatch
/// from different threads is unsafe even when the calls are serialized.
static thread_local bool tl_force_cpu = false;

/// Dispatch the forward WHT rotation to the GPU via Metal kernel.
/// The matrix is reshaped into contiguous blocks of block_size elements,
/// each processed by one threadgroup using shared memory for the butterfly.
static mlx::core::array apply_wht_rotation_gpu(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int total_blocks = (rows * cols) / bs;

    auto gpu_stream = mlx::core::default_stream(mlx::core::Device::gpu);

    // Pack kernel parameters as MLX arrays for buffer binding
    std::vector<uint32_t> bs_vec = {block_size};
    auto bs_arr = mlx::core::array(bs_vec.data(), {1}, mlx::core::uint32);

    std::vector<uint32_t> seed_vec = {seed};
    auto seed_arr = mlx::core::array(seed_vec.data(), {1}, mlx::core::uint32);

    // Flatten the matrix so blocks are contiguous in memory
    auto flat = mlx::core::reshape(matrix, {rows * cols});

    // Define TQ_WHT_BLOCK_SIZE as a preprocessor macro in the header so the
    // kernel body can size its threadgroup shared buffer. MLX's metal_kernel
    // places the header before the kernel function, making it suitable for
    // macro definitions but not threadgroup declarations. The trailing newline
    // is required to prevent the macro from consuming the next source line.
    std::string header = "#define TQ_WHT_BLOCK_SIZE " + std::to_string(bs) + "\n";

    // The kernel cache is shared across threads. A mutex guards concurrent
    // access to prevent data races when the converter's parallel workers
    // call apply_wht_rotation_gpu simultaneously with different block sizes.
    static std::mutex cache_mutex;
    static std::unordered_map<int, mlx::core::fast::CustomKernelFunction> kernel_cache;

    mlx::core::fast::CustomKernelFunction* kernel_ptr;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = kernel_cache.find(bs);
        if (it == kernel_cache.end()) {
            auto kernel = mlx::core::fast::metal_kernel(
                "tq_wht_forward_" + std::to_string(bs),
                {"matrix", "block_size_param", "seed_param"},
                {"output"},
                kWhtForwardKernelSource,
                header,
                true);
            it = kernel_cache.emplace(bs, std::move(kernel)).first;
        }
        kernel_ptr = &it->second;
    }

    auto results = (*kernel_ptr)(
        {flat, bs_arr, seed_arr},
        {{rows * cols}},
        {mlx::core::float32},
        std::make_tuple(total_blocks * bs, 1, 1),  // total threads
        std::make_tuple(bs, 1, 1),                   // threads per threadgroup
        {},
        std::nullopt,
        false,
        gpu_stream);

    // Force synchronous evaluation so GPU work completes before returning.
    // This prevents Metal command buffer conflicts when the converter's parallel
    // workers call WHT rotation from different threads under a shared mutex.
    auto output = mlx::core::reshape(results[0], {rows, cols});
    mlx::core::eval(output);
    return output;
}

/// Dispatch the inverse WHT rotation to the GPU via Metal kernel.
static mlx::core::array apply_inverse_wht_rotation_gpu(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int total_blocks = (rows * cols) / bs;

    auto gpu_stream = mlx::core::default_stream(mlx::core::Device::gpu);

    std::vector<uint32_t> bs_vec = {block_size};
    auto bs_arr = mlx::core::array(bs_vec.data(), {1}, mlx::core::uint32);

    std::vector<uint32_t> seed_vec = {seed};
    auto seed_arr = mlx::core::array(seed_vec.data(), {1}, mlx::core::uint32);

    auto flat = mlx::core::reshape(matrix, {rows * cols});

    std::string header = "#define TQ_WHT_BLOCK_SIZE " + std::to_string(bs) + "\n";

    static std::mutex inv_cache_mutex;
    static std::unordered_map<int, mlx::core::fast::CustomKernelFunction> inv_kernel_cache;

    mlx::core::fast::CustomKernelFunction* inv_kernel_ptr;
    {
        std::lock_guard<std::mutex> lock(inv_cache_mutex);
        auto it = inv_kernel_cache.find(bs);
        if (it == inv_kernel_cache.end()) {
            auto kernel = mlx::core::fast::metal_kernel(
                "tq_wht_inverse_" + std::to_string(bs),
                {"matrix", "block_size_param", "seed_param"},
                {"output"},
                kWhtInverseKernelSource,
                header,
                true);
            it = inv_kernel_cache.emplace(bs, std::move(kernel)).first;
        }
        inv_kernel_ptr = &it->second;
    }

    auto results = (*inv_kernel_ptr)(
        {flat, bs_arr, seed_arr},
        {{rows * cols}},
        {mlx::core::float32},
        std::make_tuple(total_blocks * bs, 1, 1),
        std::make_tuple(bs, 1, 1),
        {},
        std::nullopt,
        false,
        gpu_stream);

    auto output = mlx::core::reshape(results[0], {rows, cols});
    mlx::core::eval(output);
    return output;
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// CPU forward WHT: sign multiply -> butterfly FWHT -> scale by 1/sqrt(n).
/// Used for small matrices or when GPU is unavailable.
static mlx::core::array apply_wht_rotation_cpu(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    mlx::core::eval(matrix);
    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int num_full = cols / bs;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(bs));
    const auto signs = make_signs(seed, block_size);

    std::vector<float> buf(rows * cols);
    const float* src = matrix.data<float>();
    std::copy(src, src + rows * cols, buf.data());

    for (int r = 0; r < rows; ++r) {
        float* row = buf.data() + r * cols;
        for (int b = 0; b < num_full; ++b) {
            float* blk = row + b * bs;
            for (int j = 0; j < bs; ++j) blk[j] *= signs[j];
            fwht_inplace(blk, bs);
            for (int j = 0; j < bs; ++j) blk[j] *= inv_sqrt;
        }
    }

    return mlx::core::array(buf.data(), {rows, cols}, mlx::core::float32);
}

/// CPU inverse WHT: butterfly FWHT -> scale by 1/sqrt(n) -> sign multiply.
/// Used for small matrices or when GPU is unavailable.
static mlx::core::array apply_inverse_wht_rotation_cpu(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    mlx::core::eval(matrix);
    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int num_full = cols / bs;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(bs));
    const auto signs = make_signs(seed, block_size);

    std::vector<float> buf(rows * cols);
    const float* src = matrix.data<float>();
    std::copy(src, src + rows * cols, buf.data());

    for (int r = 0; r < rows; ++r) {
        float* row = buf.data() + r * cols;
        for (int b = 0; b < num_full; ++b) {
            float* blk = row + b * bs;
            fwht_inplace(blk, bs);
            for (int j = 0; j < bs; ++j) blk[j] *= inv_sqrt * signs[j];
        }
    }

    return mlx::core::array(buf.data(), {rows, cols}, mlx::core::float32);
}

// ---------------------------------------------------------------------------
// Public API: routes to GPU or CPU based on matrix size
// ---------------------------------------------------------------------------

void set_force_cpu(bool force_cpu) {
    tl_force_cpu = force_cpu;
}

bool get_force_cpu() {
    return tl_force_cpu;
}

mlx::core::array apply_wht_rotation(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int remainder = cols % bs;

    assert(remainder == 0 &&
           "in_features must be divisible by block_size — use adaptive block_size");

    // Route to GPU for large matrices where Metal dispatch overhead is amortized,
    // block_size fits within Metal's 1024-thread threadgroup limit, and the
    // thread-local CPU-only flag is not set (concurrent GPU dispatch is unsafe).
    if (!tl_force_cpu && rows * cols >= kGpuDispatchThreshold && bs <= 1024) {
        return apply_wht_rotation_gpu(matrix, seed, block_size);
    }
    return apply_wht_rotation_cpu(matrix, seed, block_size);
}

mlx::core::array apply_inverse_wht_rotation(
    const mlx::core::array& matrix,
    uint32_t seed,
    uint32_t block_size) {

    const int rows = static_cast<int>(matrix.shape(0));
    const int cols = static_cast<int>(matrix.shape(1));
    const int bs = static_cast<int>(block_size);
    const int remainder = cols % bs;

    assert(remainder == 0 &&
           "in_features must be divisible by block_size — use adaptive block_size");

    if (!tl_force_cpu && rows * cols >= kGpuDispatchThreshold && bs <= 1024) {
        return apply_inverse_wht_rotation_gpu(matrix, seed, block_size);
    }
    return apply_inverse_wht_rotation_cpu(matrix, seed, block_size);
}

} // namespace turboquant
