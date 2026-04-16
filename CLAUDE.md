# TurboQuant-MLX Core — Agent Context

## Project Overview

Core C++ library implementing TurboQuant weight and KV cache compression
for Apple Silicon via MLX. Produces `libturboquant_mlx.dylib` and
`turboquant_mlx.metallib`. Consumed by SwiftLM fork via C API bridge.

## Build Commands

```bash
# CMake build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# SPM build
swift build -c release
swift test

# With benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_BENCHMARKS=ON
cmake --build build
./build/tq_benchmarks
```

## Coding Conventions

- C++17, namespace `turboquant`
- Headers in `include/turboquant/`, sources in `src/`
- Metal kernels in `metal/`, named `turboquant_*.metal`
- Use MLX array types (`mlx::core::array`) for all tensor operations
- C API in `include/turboquant_c/` — pure C, no C++ types in signatures
- Comments: professional, enterprise-grade, describe intent not mechanics
- No aspirational comments — never describe behavior that isn't implemented

## Cleanroom Policy

This project is MIT-licensed. To maintain clean IP:

**NEVER read, copy, or reference code from:**
- `arozanov/turboquant-mlx` (Apache-2.0)
- `TheTom/turboquant_plus` (Apache-2.0)

**Safe to reference freely (MIT):**
- `cksac/turboquant-model` — Lloyd-Max codebook math, rotation logic
- `Hmbown/ZMLX` — C++ MLX extension patterns
- `TheTom/llama-cpp-turboquant` — C WHT implementation, Lloyd-Max tables
- `helgklaizar/turboquant-mlx` — Metal kernel patterns, KV cache
- `tonbistudio/turboquant-pytorch` — Full TurboQuant Python reference
- `ninfueng/lloyd-max-quantizer` — Lloyd-Max quantizer

**Safe to implement from (public academic work):**
- TurboQuant paper (Zandieh et al., ICLR 2026)
- MLX documentation and examples
- Standard signal processing and GPU programming references

## Architecture

```
include/turboquant/  → Public C++ API headers
include/turboquant_c/ → C API for Swift bridge
src/                 → C++ implementations
metal/               → Metal shader programs
tools/               → CLI tools (tq-convert)
tests/unit/          → Per-component unit tests
tests/integration/   → Cross-component integration tests
tests/c_api/         → Pure C boundary tests (validates Swift bridge)
benchmarks/          → Performance measurement
bindings/            → Python (nanobind) and Node.js (N-API) wrappers
```

## Key Files

- `codebook.h/cpp` — Lloyd-Max codebook generation
- `quantizer.h/cpp` + `rotation.cpp` — Offline weight quantization with WHT
- `linear.h/cpp` — TurboQuantLinear module (replaces nn::Linear)
- `kv_cache.h/cpp` + `decode_buffer.h/cpp` — Online KV cache compression
- `turboquant_c.h/cpp` — C API consumed by SwiftLM
- `turboquant_dequant_mm.metal` — Performance-critical fused kernel

## Metal Kernel Development Workflow

New kernels follow a four-stage pipeline:

1. **Prototype via `mx.fast.metal_kernel` (Python inline source)**
   Fast iteration, no build step. Validate correctness against CPU reference.
   CRITICAL: grid parameter is **total threads** (dispatchThreads semantics),
   NOT threadgroup count. For N threadgroups of T threads, pass grid=(N*T, ...).

2. **Validate correctness against CPU reference implementation**
   The CPU dequantizer (`dequantize_weight_cpu`) is the ground truth. Max
   absolute error should be < 1e-3 for float32, < 0.05 for float16 accumulation.

3. **Port to .metal files in the C++ extension**
   Full Metal API with explicit `dispatchThreads` or `dispatchThreadgroups`
   semantics. Compiled into `turboquant_mlx.metallib` via CMake's
   `mlx_build_metallib()`. This is the shipping path.

4. **Benchmark only the C++ version**
   Python prototype numbers are meaningless for performance. The C++ extension
   path has access to threadgroup memory barriers, simdgroup operations, and
   Metal compiler optimizations that `mx.fast.metal_kernel` cannot expose.

The Python inline path is a scratchpad. Every kernel that ships goes through
the C++ extension build.

## Testing

Unit tests validate individual components. Integration tests validate
component interaction. C API tests are written in pure C (not C++) to
guarantee the Swift interop boundary is clean.

Test fixtures in `tests/fixtures/` contain a tiny 2-layer model for
fast testing without downloading real models.
