# TurboQuant-MLX Model Conversion

TurboQuant-MLX ships two CLI tools for producing and maintaining TurboQuant-compressed model directories: `tq-convert` runs the full conversion from a HuggingFace source model, and `tq-emit-sidecar` regenerates the distributed-loader sidecar for a model that has already been converted.

Both tools are built by the top-level CMake target and end up in `build/` alongside `libturboquant_mlx.dylib` and the Metal library.

## Tools

### tq-convert — convert a source model to TurboQuant format

```
tq-convert --model <source_dir> [--output <dest_dir>] [--bits N]
          [--residual-bits N] [--block-size N] [--sensitive-layers N]
          [--per-layer-codebooks]
```

Inputs: a HuggingFace-format source model directory containing `config.json`, one or more `*.safetensors` shards, and tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `merges.txt`, `vocab.json`).

Behavior: runs the TurboQuant Lloyd-Max codebook optimization on every 2D `.weight` tensor, writes TQ-encoded safetensors to `<dest_dir>`, copies tokenizer and configuration files verbatim, injects a `quantization_config` block into `config.json`, and writes the `tq_shard_metadata.json` sidecar consumed by distributed-inference loaders.

Example:

```bash
tq-convert --model ~/turboquant-weights/source/Qwen2.5-Coder-3B \
           --output ~/turboquant-weights/converted/Qwen2.5-Coder-3B-TQ8 \
           --bits 4 --residual-bits 4
```

The `tq` dtype tag in the sidecar reflects the total primary + residual bit budget — `--bits 4 --residual-bits 4` produces `tq8` entries.

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
  "format_version": 1,
  "model_architecture": "qwen2",
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

MoE models also carry `num_experts` and `top_k` at the document root.

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
