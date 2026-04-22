#include "turboquant/convert/shard_strategy.h"

namespace turboquant::convert {

std::string infer_shard_strategy(const std::string& tensor_name) {
    if (tensor_name.find("q_proj.weight") != std::string::npos ||
        tensor_name.find("k_proj.weight") != std::string::npos ||
        tensor_name.find("v_proj.weight") != std::string::npos ||
        tensor_name.find("qkv_proj.weight") != std::string::npos ||
        tensor_name.find("gate_proj.weight") != std::string::npos ||
        tensor_name.find("up_proj.weight") != std::string::npos ||
        tensor_name.find("gate_up_proj.weight") != std::string::npos) {
        return "column_parallel";
    }
    if (tensor_name.find("o_proj.weight") != std::string::npos ||
        tensor_name.find("down_proj.weight") != std::string::npos) {
        return "row_parallel";
    }
    return "replicated";
}

int32_t infer_shard_axis(const std::string& strategy) {
    if (strategy == "column_parallel") return 0;
    if (strategy == "row_parallel") return 1;
    return -1;
}

}  // namespace turboquant::convert
