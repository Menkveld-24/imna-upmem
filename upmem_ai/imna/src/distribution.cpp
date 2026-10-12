#include "distribution.h"
#include <algorithm>
#include <stdexcept>

static int32_t* find_input_addr(
    const struct DpuTensor *input,
    const struct Context *context,
    uint16_t batch_id,
    uint16_t input_channel_id
) {
    // Incrementing an int32_t* with 1 advances it by 4 bytes
    return ((batch_id * context->input.channels * context->input.bytes_per_channel) >> 2) +
           ((input_channel_id * context->input.bytes_per_channel) >> 2) + input->data_ptr<int32_t>();
    // return ((batch_id * context->input.channels * context->input.bytes_per_channel) >> 2) +
    //        ((input_channel_id * context->input.bytes_per_channel) >> 2) + context->input_base_addr;
}

static int32_t* find_output_addr(
    const struct Context *context,
    uint16_t batch_id,
    uint16_t output_channel_id
) {
    // Incrementing an int32_t* with 1 advances it by 4 bytes
    return ((batch_id * context->output.channels * context->output.bytes_per_channel) >> 2) +
           ((output_channel_id * context->output.bytes_per_channel) >> 2) + context->output_base_addr;
}

// static int8_t* find_sign_kernel_addr(
//     const struct Context *context,
//     uint16_t output_channel_id
// ) {
//     return context->sign_base_addr + (output_channel_id * context->input.channels * context->bytes_per_kernel);
// }

// static int8_t* find_shift_kernel_addr(
//     const struct Context *context,
//     uint16_t output_channel_id
// ) {
//     return context->shift_base_addr + (output_channel_id * context->input.channels * context->bytes_per_kernel);
// }

static int8_t* find_fused_ss_kernel_addr(
    const struct Context *context,
    uint16_t output_channel_id
) {
    return context->fused_ss_base_addr + (output_channel_id * context->input.channels * context->bytes_per_kernel);
}

void createPartitioning(
    struct MramPartitioning *partitioning,
    int64_t input_offset_bytes,
    int64_t output_offset_bytes,
    int64_t weight_offset_bytes
) {
    *partitioning = {
        .input_offset_bytes  = (uint32_t) (input_offset_bytes + MULTIPLEOF8BYTES(sizeof(dpu_run_config_t))),
        .weight_offset_bytes  = (uint32_t) (weight_offset_bytes + MULTIPLEOF8BYTES(sizeof(dpu_run_config_t))),
        .output_offset_bytes   = (uint32_t) (output_offset_bytes + MULTIPLEOF8BYTES(sizeof(dpu_run_config_t))),
    };
}

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
) {
    const uint32_t DPU_MRAM_SIZE_BYTES = 62 * 1024 * 1024; // 62MB

    // Fixed MRAM cost: overhead + entire input for one batch item
    const uint32_t mram_overhead = 200
        + context->input.bytes_per_channel * context->input.channels;

    const uint32_t required_memory_size = mram_overhead +
        2 * (context->output.channels * context->input.channels * context->bytes_per_kernel)
        + (context->output.channels * context->output.bytes_per_channel)
        + (context->input.channels * context->input.bytes_per_channel);

    if (required_memory_size >= DPU_MRAM_SIZE_BYTES) {
        throw std::runtime_error("MRAM capacity will be exceeded!!");
    }

    const uint32_t total_workloads = context->batch_size;
    PRINTF("build_distribution: batch_size=%u, total_workloads=%u\n",
           context->batch_size, total_workloads);

    workloads.reserve(total_workloads);
    run_configs.reserve(total_workloads);
    configs.reserve(total_workloads);

    // const uint32_t base_cout = context->output.channels / dpus_per_batch;
    // const uint32_t extra_cout = context->output.channels % dpus_per_batch;

    uint32_t workload_id = 0;
    uint32_t cout_offset = 0;

    for (uint32_t batch = 0; batch < context->batch_size; batch++) {

        const uint32_t run_config_idx = run_configs.size();
        run_configs.push_back((dpu_run_config_t){
            .start_input_index_offset   = 0,
            .input_index_offset         = 0,
            .inputs_per_output          = (uint16_t)context->input.channels,
            .start_kernel_index_offset  = 0,
            .kernel_index_offset        = 0,
            .num_outputs                = (uint16_t)context->output.channels,
            .input_stride               = 0,
        });

        const uint32_t config_idx = configs.size();
        dpu_config_padded_t cfg = {};
        cfg.config.num_configs              = 1;
        cfg.config.num_inputs               = context->input.channels;
        cfg.config.num_outputs              = context->output.channels;
        cfg.config.num_kernels              = context->output.channels;
        cfg.config.run_config_base_bytes    = 0;
        cfg.config.input_feature_height     = context->input.height;
        cfg.config.input_feature_width      = context->input.width;
        cfg.config.output_feature_height    = context->output.height;
        cfg.config.output_feature_width     = context->output.width;
        cfg.config.opcode                   = opcode;
        cfg.config.padding                  = padding;
        cfg.config.stride                   = stride;
        cfg.config.kernel_size              = kernel_size;


        // Data order:
        // - Run configs
        // - Either input or output
        // - Shift kernels
        // - Sign kernels
        // - Either input or output

        // Input
        cfg.config.input_base_bytes = mram_partitioning->input_offset_bytes;

        // Output
        cfg.config.output_base_bytes = mram_partitioning->output_offset_bytes;

        // Shiftkernel goes at the start of the weight offset bytes
        cfg.config.fused_ss_kernel_base_bytes = mram_partitioning->weight_offset_bytes;

        // Signkernel goes after the shiftkernel
        // cfg.config.signkernel_base_bytes = cfg.config.shiftkernel_base_bytes
        //     + MULTIPLEOF8BYTES(context->output.channels * context->input.channels * context->bytes_per_kernel);

        configs.push_back(cfg);

        int32_t *input_addr = nullptr;

        if (input->onCPU()) {
            input_addr = find_input_addr(input, context, batch, 0);
        }

        workloads.push_back((dpu_workload_t){
            .workload_id      = workload_id++,
            .batch_id         = batch,
            .run_config_count = 1,
            .cout_offset      = cout_offset,
            .cout_count       = context->output.channels,
            .input_addr       = input_addr,
            .output_addr      = nullptr,
            // .shift_addr       = find_shift_kernel_addr(context, cout_offset),
            // .sign_addr        = find_sign_kernel_addr(context, cout_offset),
            .fused_ss_addr    = find_fused_ss_kernel_addr(context, cout_offset),
            .run_config_addr  = &run_configs[run_config_idx],
            .config_addr      = &configs[config_idx],
        });
    }
}


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
) {
    const uint32_t DPU_MRAM_SIZE_BYTES = 62 * 1024 * 1024; // 62MB

    // Fixed MRAM cost: overhead + entire input for one batch item
    const uint32_t mram_overhead = 200
        + context->input.bytes_per_channel * context->input.channels;

    const uint32_t required_memory_size = mram_overhead +
        2 * (context->output.channels * context->input.channels * context->bytes_per_kernel)
        + (context->output.channels * context->output.bytes_per_channel)
        + (context->input.channels * context->input.bytes_per_channel);

    if (required_memory_size >= DPU_MRAM_SIZE_BYTES) {
        throw std::runtime_error("MRAM capacity will be exceeded!!");
    }

    const uint32_t total_workloads = context->batch_size;
    PRINTF("build_distribution: batch_size=%u, total_workloads=%u\n",
           context->batch_size, total_workloads);

    workloads.reserve(total_workloads);
    run_configs.reserve(total_workloads);
    configs.reserve(total_workloads);

    // const uint32_t base_cout = context->output.channels / dpus_per_batch;
    // const uint32_t extra_cout = context->output.channels % dpus_per_batch;

    uint32_t workload_id = 0;
    uint32_t cout_offset = 0;

    for (uint32_t batch = 0; batch < context->batch_size; batch++) {

        const uint32_t run_config_idx = run_configs.size();
        run_configs.push_back((dpu_run_config_t){
            .start_input_index_offset   = 0,
            .input_index_offset         = 0,
            .inputs_per_output          = 1,
            .start_kernel_index_offset  = 0,
            .kernel_index_offset        = 0,
            .num_outputs                = (uint16_t)context->output.channels,
            .input_stride               = 1,
        });

        const uint32_t config_idx = configs.size();
        dpu_config_padded_t cfg = {};
        cfg.config.num_configs              = 1;
        cfg.config.num_inputs               = context->input.channels;
        cfg.config.num_outputs              = context->output.channels;
        cfg.config.num_kernels              = context->output.channels;
        cfg.config.run_config_base_bytes    = 0;
        cfg.config.input_feature_height     = context->input.height;
        cfg.config.input_feature_width      = context->input.width;
        cfg.config.output_feature_height    = context->output.height;
        cfg.config.output_feature_width     = context->output.width;
        cfg.config.opcode                   = opcode;
        cfg.config.padding                  = padding;
        cfg.config.stride                   = stride;
        cfg.config.kernel_size              = kernel_size;


        // Data order:
        // - Run configs
        // - Either input or output
        // - Shift kernels
        // - Sign kernels
        // - Either input or output

        // Input
        cfg.config.input_base_bytes = mram_partitioning->input_offset_bytes;

        // Output
        cfg.config.output_base_bytes = mram_partitioning->output_offset_bytes;

        // Shiftkernel goes at the start of the weight offset bytes
        cfg.config.fused_ss_kernel_base_bytes = mram_partitioning->weight_offset_bytes;

        // Signkernel goes after the shiftkernel
        // cfg.config.signkernel_base_bytes = cfg.config.shiftkernel_base_bytes
        //     + MULTIPLEOF8BYTES(context->output.channels * context->input.channels * context->bytes_per_kernel);

        configs.push_back(cfg);

        int32_t *input_addr = nullptr;

        if (input->onCPU()) {
            input_addr = find_input_addr(input, context, batch, 0);
        }

        workloads.push_back((dpu_workload_t){
            .workload_id      = workload_id++,
            .batch_id         = batch,
            .run_config_count = 1,
            .cout_offset      = cout_offset,
            .cout_count       = context->output.channels,
            .input_addr       = input_addr,
            .output_addr      = nullptr,
            .fused_ss_addr    = nullptr,
            .run_config_addr  = &run_configs[run_config_idx],
            .config_addr      = &configs[config_idx],
        });
    }
}