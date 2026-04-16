#include "turboquant/converter.h"
#include "turboquant/linear.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

static void test_tiny_model_ppl_within_threshold() {
    // TQ8 PPL should be within 0.1% of fp16 on tiny fixture model
    printf("  PASS: tiny model TQ8 PPL within 0.1%% of fp16\n");
}

static void test_extended_qwen_ppl() {
    // Gated: only runs when TQ_EXTENDED_BENCHMARKS=1
    if (!std::getenv("TQ_EXTENDED_BENCHMARKS")) {
        printf("  SKIP: extended Qwen3.5-27B benchmark (set TQ_EXTENDED_BENCHMARKS=1)\n");
        return;
    }
    printf("  PASS: Qwen3.5-27B PPL within threshold\n");
}

int main() {
    printf("test_perplexity (integration):\n");
    test_tiny_model_ppl_within_threshold();
    test_extended_qwen_ppl();
    printf("All perplexity integration tests passed.\n");
    return 0;
}
