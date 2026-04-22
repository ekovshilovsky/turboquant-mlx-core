#include "turboquant/convert/emit_sidecar.h"
#include "turboquant/turboquant.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

static void print_usage() {
    printf("Usage: tq-emit-sidecar <converted_dir>\n");
    printf("\n");
    printf("Generate tq_shard_metadata.json for an already-TurboQuant-converted model.\n");
    printf("\n");
    printf("Inspects every *.safetensors file in <converted_dir> and writes a sidecar\n");
    printf("describing per-tensor shape, dtype, byte window, and shard strategy. Used\n");
    printf("when a model was converted before the sidecar format existed or when the\n");
    printf("sidecar needs to be regenerated without re-running the full conversion.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --help, -h     Show this message\n");
    printf("  --version      Print version and exit\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string converted_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--version") {
            printf("tq-emit-sidecar %s\n", turboquant::version());
            return 0;
        }
        if (arg.rfind("--", 0) == 0) {
            fprintf(stderr, "Error: unknown option %s\n", arg.c_str());
            print_usage();
            return 1;
        }
        if (converted_dir.empty()) {
            converted_dir = arg;
        } else {
            fprintf(stderr, "Error: unexpected extra argument %s\n", arg.c_str());
            print_usage();
            return 1;
        }
    }

    if (converted_dir.empty()) {
        fprintf(stderr, "Error: missing <converted_dir> argument\n");
        print_usage();
        return 1;
    }

    const auto result = turboquant::convert::emit_sidecar_for_directory(
        std::filesystem::path(converted_dir));
    if (!result.ok) {
        fprintf(stderr, "Error: %s\n", result.error.c_str());
        return 1;
    }

    printf("Wrote %s\n", result.sidecar_path.string().c_str());
    printf("  tensors:         %d\n", result.tensor_count);
    printf("  column_parallel: %d\n", result.column_parallel_count);
    printf("  row_parallel:    %d\n", result.row_parallel_count);
    printf("  replicated:      %d\n", result.replicated_count);
    printf("  expert_parallel: %d\n", result.expert_parallel_count);
    return 0;
}
