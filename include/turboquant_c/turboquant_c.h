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

/** Get library version string. */
const char* tq_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_C_H */
