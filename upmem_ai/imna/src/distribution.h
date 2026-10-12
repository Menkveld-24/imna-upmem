#pragma once
#include <vector>
#include "dpu_dtypes.h"
#include "dputensor.h"

// void build_distribution(
//     const struct Context *context,
//     DpuTensor *input,
//     std::vector<dpu_workload_t> &workloads,
//     std::vector<dpu_run_config_t> &run_configs,
//     std::vector<dpu_config_padded_t> &configs
// );

// Pre-computed MRAM offsets for multi-layer chaining.
// All offsets are in bytes relative to the start of mram_arena.
struct MramPartitioning {
    uint32_t input_offset_bytes;
    uint32_t weight_offset_bytes;
    uint32_t output_offset_bytes;
};

void build_distribution(
    const struct Context *context,
    const struct DpuTensor *input,
    const struct MramPartitioning *mram_partitioning,
    std::vector<dpu_workload_t> &workloads,
    std::vector<dpu_run_config_t> &run_configs,
    std::vector<dpu_config_padded_t> &configs,
    uint8_t opcode,
    uint8_t padding,
    uint8_t kernel_size,
    uint8_t stride
);

void build_pooling_distribution(
    const struct Context *context,
    const struct DpuTensor *input,
    const struct MramPartitioning *mram_partitioning,
    std::vector<dpu_workload_t> &workloads,
    std::vector<dpu_run_config_t> &run_configs,
    std::vector<dpu_config_padded_t> &configs,
    uint8_t opcode,
    uint8_t padding,
    uint8_t kernel_size,
    uint8_t stride
);

void createPartitioning(
    struct MramPartitioning *partitioning,
    int64_t input_offset_bytes,
    int64_t output_offset_bytes,
    int64_t weight_offset_bytes
);