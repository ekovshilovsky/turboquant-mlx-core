# Rust Bindings for TurboQuant-MLX

Rust bindings are planned but not yet implemented. The C API (`turboquant_c.h`) provides the stable FFI surface that Rust will bind against.

## Quick Start (when implemented)

```rust
use turboquant_mlx::*;

let version = tq_version();
let model = tq_model_load("/path/to/tq-model")?;
```

## How to Generate

The bindings can be auto-generated from the C header using `bindgen`:

```bash
cargo install bindgen-cli
bindgen include/turboquant_c/turboquant_c.h -o bindings/rust/src/ffi.rs
```

Then wrap the raw FFI in safe Rust types following the same pattern as the Swift and Python bindings.

## Contributing

If you'd like to build the Rust bindings, open an issue or PR. The C API is stable and the binding generation is mechanical — the main work is designing the safe Rust wrapper types.
