#include <cstdio>

// Forward declarations for benchmark suites
void bench_dequant();
void bench_linear();
void bench_distributed();
void bench_e2e();

int main(int argc, char* argv[]) {
    printf("TurboQuant-MLX Benchmarks\n");
    printf("========================\n\n");

    bench_dequant();
    bench_linear();
    bench_distributed();
    bench_e2e();

    return 0;
}
