#include <metal_stdlib>
using namespace metal;

/// Quantized Johnson-Lindenstrauss (QJL) 1-bit residual compression for K-cache.
///
/// After PolarQuant quantization, a residual remains between the original
/// key vector and its dequantized approximation. QJL compresses this residual
/// using random projection followed by sign extraction, storing only 1 bit
/// per projected dimension plus a scalar norm.
///
/// The correction term during attention scoring is:
///   correction = sqrt(pi/2) / d * dot(signs_q, signs_k) * norm_k
///
/// Note: V3 results from tonbistudio/turboquant-pytorch indicate that QJL
/// variance can degrade softmax quality. The compress function is provided
/// for experimentation; the correction may be omitted in production configs
/// where K-cache PolarQuant alone provides sufficient accuracy.
///
/// tq_qjl_compress layout:
///   One threadgroup per vector (per head, per token).
///   grid.x  = num_heads
///   grid.y  = num_tokens
///   threads_per_threadgroup = proj_dim (number of random projections, power of 2)
///
/// Buffer layout for tq_qjl_compress:
///   buffer(0) — residual    : float  [num_tokens, num_heads, head_dim]
///   buffer(1) — params      : uint32 [4] {head_dim, num_heads, proj_dim, proj_seed}
///   buffer(2) — signs_out   : uint32 [num_tokens * num_heads * ceil(proj_dim/32)]
///   buffer(3) — norms_out   : float  [num_tokens * num_heads]

constant uint MAX_PROJ_DIM = 1024;

// ---------------------------------------------------------------------------
// QJL random projection: pseudo-random +/-1 matrix entries from a seed.
// Uses a hash combining projection row index, column index, and seed to
// produce a deterministic sign. Independent of the WHT sign hash.
// ---------------------------------------------------------------------------

inline float qjl_proj_sign(uint seed, uint proj_row, uint col) {
    uint h = seed ^ (proj_row * 2654435761u) ^ (col * 2246822519u);
    h ^= (h >> 16);
    h *= 0x45d9f3bu;
    return (h & 1u) ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// QJL compress kernel: random projection + sign extraction + norm
// ---------------------------------------------------------------------------

kernel void tq_qjl_compress(
    device const float*    residual   [[buffer(0)]],
    device const uint32_t* params     [[buffer(1)]],
    device uint32_t*       signs_out  [[buffer(2)]],
    device float*          norms_out  [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint head_dim  = params[0];
    const uint num_heads = params[1];
    const uint proj_dim  = params[2];
    const uint proj_seed = params[3];

    const uint head_idx  = gid.x;
    const uint token_idx = gid.y;

    if (tid >= proj_dim) return;

    // Base offset into the residual vector for this (token, head).
    const uint vec_offset = token_idx * num_heads * head_dim
                          + head_idx * head_dim;

    // ---- Step 1: Compute random projection dot product for this row ----
    // Each thread computes one row of the random projection matrix dotted
    // with the residual vector. The projection matrix S has entries +/-1.

    float dot = 0.0f;
    for (uint c = 0; c < head_dim; ++c) {
        float sign = qjl_proj_sign(proj_seed, tid, c);
        dot += sign * residual[vec_offset + c];
    }

    // ---- Step 2: Compute L2 norm of residual via parallel reduction ----

    threadgroup float shared_buf[MAX_PROJ_DIM];

    // First thread of each group contributes to norm. Only threads with
    // tid < head_dim load squared residual values; others contribute zero.
    if (tid < head_dim) {
        float r = residual[vec_offset + tid];
        shared_buf[tid] = r * r;
    } else {
        shared_buf[tid] = 0.0f;
    }

    // Reduce over max(head_dim, proj_dim) to handle both dimensions.
    uint reduce_n = (head_dim > proj_dim) ? proj_dim : head_dim;
    // Ensure we reduce over at least proj_dim entries (padded with zeros).
    reduce_n = proj_dim;
    for (uint stride = reduce_n / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float l2_norm = sqrt(shared_buf[0]);

    if (tid == 0) {
        norms_out[token_idx * num_heads + head_idx] = l2_norm;
    }

    // ---- Step 3: Extract sign bit and pack into uint32 words ----

    uint sign_bit = (dot >= 0.0f) ? 1u : 0u;

    // Pack 32 sign bits per uint32. Thread tid belongs to word (tid / 32)
    // at bit position (tid % 32).
    threadgroup uint shared_signs[MAX_PROJ_DIM];
    shared_signs[tid] = sign_bit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint word_idx    = tid / 32;
    const uint bit_pos     = tid % 32;
    const uint words_per_vector = (proj_dim + 31) / 32;

    if (bit_pos == 0 && word_idx < words_per_vector) {
        uint packed = 0;
        uint base = word_idx * 32;
        for (uint i = 0; i < 32 && (base + i) < proj_dim; ++i) {
            packed |= (shared_signs[base + i] & 1u) << i;
        }
        uint out_offset = (token_idx * num_heads + head_idx) * words_per_vector
                        + word_idx;
        signs_out[out_offset] = packed;
    }
}

// ---------------------------------------------------------------------------
// QJL dot product correction: computes the inner-product correction term
// between a query's QJL signs and a key's QJL signs.
//
// correction = sqrt(pi/2) / proj_dim * popcount(signs_q XNOR signs_k) * norm_k
//            = sqrt(pi/2) / proj_dim * (2 * match_count - proj_dim) * norm_k
//
// Threadgroup layout:
//   One threadgroup per (query, key) pair.
//   grid.x  = num_keys
//   grid.y  = num_queries (batch)
//   threads_per_threadgroup = ceil(proj_dim / 32)
//
// Buffer layout:
//   buffer(0) — signs_q    : uint32 [num_queries * num_heads * words_per_vec]
//   buffer(1) — signs_k    : uint32 [num_keys * num_heads * words_per_vec]
//   buffer(2) — norms_k    : float  [num_keys * num_heads]
//   buffer(3) — params     : uint32 [3] {proj_dim, num_heads, head_idx}
//   buffer(4) — corrections: float  [num_queries * num_keys]
// ---------------------------------------------------------------------------

constant uint MAX_SIGN_WORDS = 32; // proj_dim / 32, supports up to 1024

kernel void tq_qjl_dot_correction(
    device const uint32_t* signs_q     [[buffer(0)]],
    device const uint32_t* signs_k     [[buffer(1)]],
    device const float*    norms_k     [[buffer(2)]],
    device const uint32_t* params      [[buffer(3)]],
    device float*          corrections [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint proj_dim  = params[0];
    const uint num_heads = params[1];
    const uint head_idx  = params[2];

    const uint key_idx   = gid.x;
    const uint query_idx = gid.y;

    const uint words_per_vec = (proj_dim + 31) / 32;

    if (tid >= words_per_vec) return;

    // Load one word from query signs and key signs, compute XNOR popcount.
    uint q_offset = (query_idx * num_heads + head_idx) * words_per_vec + tid;
    uint k_offset = (key_idx * num_heads + head_idx) * words_per_vec + tid;

    uint q_word = signs_q[q_offset];
    uint k_word = signs_k[k_offset];

    // XNOR: matching bits indicate matching signs.
    uint xnor_val = ~(q_word ^ k_word);
    uint match_count = popcount(xnor_val);

    // Parallel reduction to sum match counts across all words.
    threadgroup uint shared_count[MAX_SIGN_WORDS];
    shared_count[tid] = match_count;

    for (uint stride = words_per_vec / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            shared_count[tid] += shared_count[tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint total_matches = shared_count[0];
        // Signed dot product estimate: (2 * matches - proj_dim) maps
        // [0, proj_dim] match count to [-proj_dim, proj_dim] inner product.
        float signed_dot = float(2 * total_matches) - float(proj_dim);

        // QJL correction factor: sqrt(pi/2) / proj_dim * signed_dot * norm_k
        const float SQRT_PI_OVER_2 = 1.2533141f;
        float norm = norms_k[key_idx * num_heads + head_idx];
        float correction = SQRT_PI_OVER_2 / float(proj_dim) * signed_dot * norm;

        corrections[query_idx * gid.x + key_idx] = correction;
    }
}
