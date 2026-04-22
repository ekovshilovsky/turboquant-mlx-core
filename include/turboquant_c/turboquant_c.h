#ifndef TURBOQUANT_C_H
#define TURBOQUANT_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TurboQuant C API — stable interface for Swift and other language bindings.
 * All functions are safe to call from C code. Memory management follows
 * create/free pairing: every tq_*_create has a matching tq_*_free.
 */

/* Opaque handle types */
typedef struct tq_model_s* tq_model_t;
typedef struct tq_kv_cache_s* tq_kv_cache_t;
typedef struct tq_coordinator_s* tq_coordinator_t;
typedef struct tq_cluster_s* tq_cluster_t;
typedef struct tq_linear_s* tq_linear_t;

/* -- Model API ----------------------------------------------------------- */

/** Load a TurboQuant-compressed model from the given directory. Returns NULL on failure. */
tq_model_t tq_model_load(const char* model_path);

/** Run forward pass. Caller must free the returned array via tq_array_free. */
void* tq_model_forward(tq_model_t model, const void* input_array);

/** Free model resources. Safe to call with NULL. */
void tq_model_free(tq_model_t model);

/* -- KV Cache API -------------------------------------------------------- */

/** Create a TQ-compressed KV cache. */
tq_kv_cache_t tq_kv_cache_create(
    int num_layers,
    int num_heads,
    int head_dim,
    int kv_bits,
    int max_context,
    int decode_window);

/** Append new KV entries for the specified layer. */
void tq_kv_cache_append(
    tq_kv_cache_t cache,
    int layer,
    const void* keys_array,
    const void* values_array);

/** Get fp16 keys from decode window. Caller must free via tq_array_free. */
void* tq_kv_cache_get_keys_fp16(tq_kv_cache_t cache, int layer, int start, int end);

/** Get fp16 values from decode window. Caller must free via tq_array_free. */
void* tq_kv_cache_get_values_fp16(tq_kv_cache_t cache, int layer, int start, int end);

/** Run fused TQ attention on compressed region. Caller must free via tq_array_free. */
void* tq_kv_cache_attention(
    tq_kv_cache_t cache,
    int layer,
    const void* queries_array,
    int compressed_start,
    int compressed_end);

/** Current sequence length in the cache. */
int tq_kv_cache_seq_length(tq_kv_cache_t cache);

/** Free KV cache resources. Safe to call with NULL. */
void tq_kv_cache_free(tq_kv_cache_t cache);

/* -- Distributed API ----------------------------------------------------- */

/** Initialize distributed coordinator from hostfile. Backend: "jaccl", "ring", "auto". */
tq_coordinator_t tq_distributed_init(const char* hostfile, const char* backend);

/** Initialize for single-node inference. */
tq_coordinator_t tq_distributed_init_local(void);

/** Get this node's rank. */
int tq_distributed_rank(tq_coordinator_t coord);

/** Get cluster world size. */
int tq_distributed_world_size(tq_coordinator_t coord);

/** Free coordinator resources. Safe to call with NULL. */
void tq_distributed_free(tq_coordinator_t coord);

/* -- Cluster API --------------------------------------------------------- */

/** Begin building a cluster plan for a model with the given dimensions.
 *  Returns NULL on failure (e.g., non-positive dimensions). */
tq_cluster_t tq_cluster_create(int num_layers, int num_heads, int head_dim);

/** Register a discovered peer with its usable memory in GB.
 *  Returns 0 on success, -1 on failure (e.g., NULL cluster, NULL hostname,
 *  non-positive memory, or planning already performed). Peers are assigned
 *  rank values in insertion order starting at 0. */
int tq_cluster_add_node(tq_cluster_t cluster, const char* hostname, double memory_gb);

/** Compute the memory-aware pipeline-parallel shard plan across all
 *  previously-added nodes. Must be called exactly once; subsequent calls
 *  return -1. Returns 0 on success, -1 on failure (NULL cluster, no nodes
 *  added, or already planned). */
int tq_cluster_plan(tq_cluster_t cluster);

/** Number of nodes registered. Valid before or after tq_cluster_plan. */
int tq_cluster_node_count(tq_cluster_t cluster);

/** Inclusive start of the layer range assigned to the given rank. Returns
 *  -1 before tq_cluster_plan has been called or if rank is out of range. */
int tq_cluster_get_layer_start(tq_cluster_t cluster, int rank);

/** Exclusive end of the layer range assigned to the given rank. Returns
 *  -1 before tq_cluster_plan has been called or if rank is out of range. */
int tq_cluster_get_layer_end(tq_cluster_t cluster, int rank);

/** Release cluster resources. Safe to call with NULL. */
void tq_cluster_free(tq_cluster_t cluster);

/* -- Shard-aware Linear API ---------------------------------------------- */

/**
 * Construct a rank-local TurboQuant linear layer from pre-sharded compressed
 * weight tensors. The three tensor pointers (packed_primary, packed_residual,
 * norms) are owned by the caller and must remain valid for the duration of
 * the tq_linear_t handle: the underlying mlx::core::array wrappers are
 * reconstructed by reference.
 *
 * For column-parallel shards pass local_in_features == full_in_features
 * and supply this rank's slice of the OUTPUT dim: packed_primary and
 * packed_residual must be [rank_out_features, full_in_features / 2] (4-bit
 * packed) or [rank_out_features, full_in_features] (5-bit), and norms must
 * be [rank_out_features]. Seeds (seed_primary, seed_residual) are layer-
 * global and identical on every rank.
 *
 * For row-parallel shards pass local_in_features = full_in_features /
 * N_ranks (must respect the 9a.1 group-alignment constraint:
 * full_in_features % (N_ranks * block_size) == 0). Supply this rank's slice
 * of the INPUT dim: packed_primary and packed_residual must be
 * [out_features, local_in_features / 2] (4-bit) or [out_features,
 * local_in_features] (5-bit). norms is broadcast unchanged across ranks.
 * Seeds are also layer-global.
 *
 * packed_primary pointer type: const uint8_t*
 * packed_residual pointer type: const uint8_t* (may be zero-filled buffer
 *     when the layer has no residual codebook — same shape as primary's
 *     packed layout in 4-bit form).
 * norms pointer type: const float*
 * primary_codebook / residual_codebook pointer type: const float*, length
 *     2^primary_bits and 2^residual_bits respectively (typically 16 for
 *     4-bit, 32 for 5-bit).
 *
 * Returns NULL on allocation failure or invalid arguments (non-positive
 * dimensions, NULL packed_primary, NULL norms, NULL primary_codebook,
 * primary_bits outside {4, 5}).
 */
tq_linear_t tq_linear_create_shard(
    int full_in_features,
    int local_in_features,
    int rank_out_features,
    int primary_bits,
    int residual_bits,
    const void* packed_primary,
    const void* packed_residual,
    const void* norms,
    const void* primary_codebook,
    const void* residual_codebook,
    unsigned int seed_primary,
    unsigned int seed_residual,
    int block_size);

/**
 * Forward pass through the shard. input_array is an mlx::core::array*
 * cast to void*, with shape [batch, local_in_features] and any floating
 * dtype (cast to float16 internally). Returns a newly allocated
 * mlx::core::array* of shape [batch, rank_out_features] in float16; the
 * caller takes ownership and must release it via tq_array_free.
 *
 * Returns NULL if layer is NULL, input_array is NULL, or the kernel
 * dispatch throws an exception at the C++ boundary.
 */
void* tq_linear_forward(tq_linear_t layer, const void* input_array);

/** Release the layer's resources. Safe to call with NULL. */
void tq_linear_free(tq_linear_t layer);

/* -- Dequantization API -------------------------------------------------- */

/**
 * Dequantize a TurboQuant model directory to fp16 safetensors.
 * Reconstructs full-precision weights using the stored codebooks, inverse WHT,
 * and norm correction. The output directory can be loaded by any standard MLX
 * or HuggingFace model loader. Returns 0 on success, non-zero on error.
 */
int tq_model_dequant(const char* tq_model_path, const char* output_path);

/* -- Utility ------------------------------------------------------------- */

/** Free an array returned by tq_model_forward or tq_kv_cache_* functions. */
void tq_array_free(void* array);

/**
 * Allocate a new float32 MLX array from a row-major [rows, cols] buffer.
 * Returns an mlx::core::array* cast to void*; release via tq_array_free.
 * Returns NULL on allocation failure or non-positive dimensions. Used by
 * C-level tests and non-Swift bindings that need to construct input
 * activations without a direct MLX dependency.
 */
void* tq_array_from_fp32(const float* data, int rows, int cols);

/**
 * Copy a float16 MLX array (as returned by tq_linear_forward or the KV
 * cache APIs) into a caller-supplied float32 buffer of at least size
 * rows * cols. out_rows / out_cols must match the array's shape. Returns
 * 0 on success, -1 on shape mismatch, -1 on NULL inputs, -1 on any
 * C++-boundary exception. The output buffer is left unmodified on error.
 */
int tq_array_copy_to_fp32(const void* array, float* out, int out_rows, int out_cols);

/** Get library version string. */
const char* tq_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_C_H */
