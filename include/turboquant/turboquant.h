#pragma once

/// TurboQuant-MLX: Near-lossless weight and KV cache compression for Apple Silicon.
/// Implements the TurboQuant algorithm (Zandieh et al., ICLR 2026) as fused
/// Metal kernels integrated with the MLX framework.

#include "codebook.h"
#include "quantizer.h"
#include "dequantizer.h"
#include "linear.h"
#include "converter.h"
#include "distributed.h"
#include "kv_cache.h"
#include "decode_buffer.h"

namespace turboquant {

/// Library version string.
const char* version();

} // namespace turboquant
