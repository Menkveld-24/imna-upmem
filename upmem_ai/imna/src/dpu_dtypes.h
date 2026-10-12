#ifndef _DPU_TYPES_H
#define _DPU_TYPES_H

#include <stdint.h>
#include <vector>


#define MULTIPLEOF8BYTES(x) (((x) + 7) & ~7)

#define DUMP_MRAM 0

#define PRINTF_ENABLED 0

#if PRINTF_ENABLED
    #include <stdio.h>
    #define PRINTF(...) printf(__VA_ARGS__)
#else
    #define PRINTF(...)
#endif

struct inout_context_t {
    uint16_t channels;
    uint16_t height;
    uint16_t width;
    uint32_t bytes_per_channel;
};

struct Context {
    // Basic information
    uint16_t batch_size;
    uint16_t kernel_size;
    uint16_t padding;
    uint16_t stride;

    // Input and output tensor metadata
    struct inout_context_t input;
    struct inout_context_t output;

    // Sizes metadata for easier address calculations
    uint32_t bytes_per_kernel;

    // Addresses for the tensors
    int32_t* input_base_addr;
    // int8_t* shift_base_addr;
    // int8_t* sign_base_addr;
    int8_t* fused_ss_base_addr;
    int32_t* output_base_addr;
};

// Mirror of DPU-side structs (must match dpu.h)
struct dpu_run_config_t {
    uint16_t start_input_index_offset;
    uint16_t input_index_offset;
    uint16_t inputs_per_output;   // how many inputs accumulate into each output (C_in for conv, 1 for maxpool)
    uint16_t start_kernel_index_offset;
    uint16_t kernel_index_offset;
    uint16_t num_outputs;         // number of output channels covered by this run config
    uint16_t input_stride;        // input base shift per output (0 = shared inputs like conv, 1 = per-channel like maxpool)
};

#define OP_CONV2D   0b00000001
#define OP_RELU     0b00000010
#define OP_MAXPOOL  0b00000100

struct dpu_config_t {
    uint16_t num_configs;
    uint16_t num_inputs;
    uint16_t num_outputs;
    uint16_t num_kernels;

    uint32_t run_config_base_bytes;
    uint32_t input_base_bytes;
    uint32_t output_base_bytes;
    uint32_t fused_ss_kernel_base_bytes;

    uint16_t input_feature_height;
    uint16_t input_feature_width;
    uint16_t output_feature_height;
    uint16_t output_feature_width;
    uint8_t opcode;
    uint8_t padding;
    uint8_t stride;
    uint8_t kernel_size;
};

struct dpu_config_padded_t {
    struct dpu_config_t config;
    uint8_t padding[
        MULTIPLEOF8BYTES(sizeof(struct dpu_config_t)) - sizeof(struct dpu_config_t)
    ];
};

struct dpu_workload_t {
    uint32_t workload_id;
    uint32_t batch_id;        // first batch item assigned to this DPU
    uint32_t run_config_count; // number of batch items packed onto this DPU
    uint32_t cout_offset;     // first output channel assigned to this DPU
    uint32_t cout_count;      // number of output channels assigned to this DPU

    int32_t *input_addr;     // host pointer to first batch item's input
    int32_t *output_addr;    // host pointer to first batch item's output slice
    // int8_t *shift_addr;       // host pointer to this DPU's shift kernel slice
    // int8_t *sign_addr;        // host pointer to this DPU's sign kernel slice
    int8_t *fused_ss_addr;    // host pointer to this DPU's fused ss kernel slice
    struct dpu_run_config_t *run_config_addr;
    struct dpu_config_padded_t *config_addr;
};

struct GatherContext {
    const struct Context *context;
    const std::vector<dpu_workload_t> &workloads;
    uint32_t *offset;
    // const std::vector<dpu_config_padded_t> &dpu_configs;
    // const std::vector<dpu_run_config_t> &dpu_run_configs;
};


#endif