# TurboQuant-MLX Model Conversion

TurboQuant-MLX ships two CLI tools for producing and maintaining TurboQuant-compressed model directories: `tq-convert` runs the full conversion from a HuggingFace source model, and `tq-emit-sidecar` regenerates the distributed-loader sidecar for a model that has already been converted.

Both tools are built by the top-level CMake target and end up in `build/` alongside `libturboquant_mlx.dylib` and the Metal library.

## Tools

### tq-convert — convert a source model to TurboQuant format

```
tq-convert --model <source_dir> [--output <dest_dir>] [--draft]
          [--target-world-size N] [--bits N] [--residual-bits N]
          [--block-size N] [--sensitive-layers N] [--no-per-layer-codebooks]
```

Inputs: a HuggingFace-format source model directory containing `config.json`, one or more `*.safetensors` shards, and tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `merges.txt`, `vocab.json`).

Behavior: runs the TurboQuant Lloyd-Max codebook optimization on every 2D `.weight` tensor, writes TQ-encoded safetensors to `<dest_dir>`, copies tokenizer and configuration files verbatim, injects a `quantization_config` block into `config.json`, and writes the `tq_shard_metadata.json` sidecar consumed by distributed-inference loaders.

#### Quality defaults

The default flags produce a release-quality snapshot — operators do not need to remember which quality switches to enable:

| Flag                       | Default | Effect                                                                  |
| -------------------------- | ------- | ----------------------------------------------------------------------- |
| `--sensitive-layers N`     | `4`     | Keep first/last N transformer layers at fp16 (≈30% perplexity reduction). |
| `--per-layer-codebooks`    | on      | Fit Lloyd-Max codebooks per layer instead of a single global codebook.  |
| `--target-world-size N`    | `2`     | Largest TP world size the snapshot must support; common 2-Mac cluster. |
| `--bits N --residual-bits N` | `4 + 4` | 8-bit-equivalent budget. Validated end-to-end for v1.                |

For development iteration, `--draft` flips the quality flags off:

```bash
tq-convert --model ~/turboquant-weights/source/Qwen2.5-Coder-3B --draft
```

`--draft` is roughly 30x faster to convert but produces 10–20x worse perplexity delta. Use it for round-trip smoke tests, regression checks, and CI; do not ship draft snapshots to users.

Example (production-quality, defaults):

```bash
tq-convert --model ~/turboquant-weights/source/Qwen2.5-Coder-3B
# → ~/turboquant-weights/source/Qwen2.5-Coder-3B-TQ8-TP2
```

The `tq` dtype tag in the sidecar reflects the total primary + residual bit budget — `--bits 4 --residual-bits 4` produces `tq8` entries.

#### Output directory naming

When `--output` is omitted, the output directory is derived from the source model basename with a `-TQ8-TP{N}` suffix and placed beside the source directory. The naming mirrors AWQ's `<model>-AWQ-INT4-G128` convention so the topology capability of the snapshot is visible in both the local directory name and any downstream HuggingFace repo name. Examples:

| Source                                              | `--target-world-size` | Output                                                     |
| --------------------------------------------------- | --------------------- | ---------------------------------------------------------- |
| `~/turboquant-weights/source/Qwen2.5-Coder-3B`      | 2 (default)           | `~/turboquant-weights/source/Qwen2.5-Coder-3B-TQ8-TP2`     |
| `~/turboquant-weights/source/Qwen2.5-Coder-3B`      | 4                     | `~/turboquant-weights/source/Qwen2.5-Coder-3B-TQ8-TP4`     |
| `~/turboquant-weights/source/Qwen2.5-Coder-3B`      | 1                     | `~/turboquant-weights/source/Qwen2.5-Coder-3B-TQ8-TP1`     |

The `TQ8` token is shorthand for "TurboQuant 8-bit-equivalent" (4 primary + 4 residual bits) and is hardcoded for v1 because that is the only configuration validated end-to-end. Pass `--output` explicitly to override the derived path.

#### Choosing `--target-world-size`

`--target-world-size` is the largest tensor-parallel world size the produced snapshot is guaranteed to support. The flag is named to align with vLLM's `tensor-parallel-size` and AWQ/GPTQ's group-size flags. It constrains per-layer block-size auto-selection so `(world_size * block_size)` divides each weight's input dimension evenly — required for row-parallel sharding. Larger values trade a small amount of per-layer quality for the ability to shard across more ranks.

| Cluster | `--target-world-size` | Notes                                                       |
| ------- | --------------------- | ----------------------------------------------------------- |
| 1 Mac   | 1                     | Max-quality single-Mac snapshot, no tensor parallelism.     |
| 2 Macs  | 2 (default)           | Reasonable per-layer quality, supports up to 2 ranks.       |
| 4 Macs  | 4                     | Slightly lower per-layer quality, supports up to 4 ranks.   |

Snapshots converted with a smaller `--target-world-size` than the runtime cluster's actual size will be rejected at SwiftLM's cluster bring-up: the loader reads `max_supported_world_size` from `tq_shard_metadata.json` and surfaces an actionable error like "this fixture supports up to 2 ranks; re-convert with --target-world-size 4".

### tq-emit-sidecar — generate the sidecar for an existing TQ-converted model

```
tq-emit-sidecar <converted_dir>
```

Reads `config.json` and every `*.safetensors` file in the directory; writes `tq_shard_metadata.json` in the same directory. Does not touch the safetensors files themselves.

Used when:

1. A model was converted before the sidecar format existed.
2. The source weights have been garbage-collected from the HuggingFace cache but the TurboQuant-converted output is preserved, so a full re-conversion is not possible.
3. The sidecar format version has been bumped and existing models need a refresh.

Example:

```bash
tq-emit-sidecar ~/turboquant-weights/converted/Qwen2.5-Coder-3B-TQ8
```

The tool reuses the exact same safetensors-header parser, shard-strategy inference, and JSON writer as the `tq-convert` post-write pass. Running it against a directory already produced by `tq-convert` produces a byte-identical sidecar.

## Sidecar contents

The sidecar is a small JSON document (one entry per logical tensor, no weight payloads) with this shape:

```json
{
  "format_version": 2,
  "model_architecture": "qwen2",
  "max_supported_world_size": 2,
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "intermediate_size": 11008,
  "tensors": {
    "model.layers.0.self_attn.q_proj.weight": {
      "shape": [2048, 2048],
      "dtype": "tq8",
      "file": "model-00001-of-00002.safetensors",
      "byte_offset": 98304,
      "byte_length": 4194304,
      "shard_axis": 0,
      "shard_strategy": "column_parallel",
      "codebook_key": "model.layers.0.self_attn.q_proj.codebook_primary",
      "rotation_key": "model.layers.0.self_attn.q_proj.seeds"
    }
  }
}
```

SwiftLM's distributed-inference loader reads this sidecar to decide, for every tensor, whether to broadcast it to all ranks (`shard_strategy = replicated`, `shard_axis = null`) or slice it along a specific axis (`column_parallel` / `row_parallel` / `expert_parallel`) without re-parsing the TurboQuant packed on-disk layout.

`max_supported_world_size` is the highest tensor-parallel world size the snapshot was built for; SwiftLM checks the runtime cluster size against this value at bring-up and refuses to load when the runtime size exceeds it. Re-convert with a larger `--target-world-size` to support larger clusters.

MoE models also carry `num_experts` and `top_k` at the document root.

Schema version 2 added the `max_supported_world_size` top-level field. Older converted directories read by `tq-emit-sidecar` will be re-emitted at version 2 with `max_supported_world_size: 1` (no tensor parallelism) when the source `config.json` does not record a `max_world_size` value.

## Shard-strategy inference rules

Both tools infer each tensor's shard strategy from its fully qualified name:

| Tensor name contains                                                                   | Strategy           | Shard axis    |
| -------------------------------------------------------------------------------------- | ------------------ | ------------- |
| `q_proj`, `k_proj`, `v_proj`, `qkv_proj`, `gate_proj`, `up_proj`, `gate_up_proj`       | `column_parallel`  | 0 (output)    |
| `o_proj`, `down_proj`                                                                  | `row_parallel`     | 1 (input)     |
| Everything else (embeddings, norms, biases, lm_head, codebooks, rotations)             | `replicated`       | null          |

Expert-parallel tensors are emitted by the MoE-aware conversion path and are not inferred from the name alone. If a model uses non-standard tensor names, the default is `replicated` — safe but suboptimal for tensor parallelism.

## Source-weight caching convention

HuggingFace's `~/.cache/huggingface/hub/` layout garbage-collects blobs aggressively. To preserve source weights across cache cleanups, download them to a durable path outside the HF cache and disable symlink mode:

```bash
mkdir -p ~/turboquant-weights/source
huggingface-cli download Qwen/Qwen2.5-Coder-3B \
    --local-dir ~/turboquant-weights/source/Qwen2.5-Coder-3B \
    --local-dir-use-symlinks False
```

Without `--local-dir-use-symlinks False`, the `huggingface-cli` download produces a directory of symlinks pointing into `~/.cache/huggingface/hub/`; the symlinks break silently the next time HuggingFace prunes the cache, and any subsequent re-conversion with `tq-convert` fails with missing input weights. The convention across this project is `~/turboquant-weights/source/` for source weights and `~/turboquant-weights/converted/` for TurboQuant output.
