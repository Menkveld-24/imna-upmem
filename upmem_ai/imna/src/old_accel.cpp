#include <ATen/ops/empty_like.h>
#include <cstdlib>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include <ATen/ATen.h>
#include <dpu>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#include <time.h>
#include "dpu_dtypes.h"
#include "dpu_pool.h"

static double ms_since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1e3
         + (now.tv_nsec - start.tv_nsec) * 1e-6;
}

#define TICK(var) struct timespec var; clock_gettime(CLOCK_MONOTONIC, &var)
#define TOCK(var, label) fprintf(stderr, "  [accel] %-30s %7.2f ms\n", label, ms_since(var))

static int32_t* legacy_find_input_addr (
    const struct Context *context,
    uint16_t batch_id,
    uint16_t input_channel_id
) {
    // Incrementing an int32_t* with 1 advances it by 4 bytes
    return ((batch_id * context->input.channels * context->input.bytes_per_channel) >> 2) +
           ((input_channel_id * context->input.bytes_per_channel) >> 2) + context->input_base_addr;
}


static int32_t* legacy_find_output_addr (
    const struct Context *context,
    uint16_t batch_id,
    uint16_t output_channel_id
) {
    // Incrementing an int32_t* with 1 advances it by 4 bytes
    return ((batch_id * context->output.channels * context->output.bytes_per_channel) >> 2) +
           ((output_channel_id * context->output.bytes_per_channel) >> 2) + context->output_base_addr;
}

static int8_t* legacy_find_sign_kernel_addr (
    const struct Context *context,
    uint16_t output_channel_id
) {
    return context->sign_base_addr + (output_channel_id * context->input.channels * context->bytes_per_kernel);
}

static int8_t* legacy_find_shift_kernel_addr (
    const struct Context *context,
    uint16_t output_channel_id
) {
    return context->shift_base_addr + (output_channel_id * context->input.channels * context->bytes_per_kernel);
}



// Builds a flat list of workloads, one per (batch_item, cout_slice).
// The caller is responsible for running them in waves of dpu_limit DPUs.
// dpus_per_batch controls how finely Cout is split per batch item;
// if 0, it defaults to as many slices as fit in MRAM (maximum parallelism).
static void legacy_build_distribution(
    const struct Context *context,
    std::vector<dpu_workload_t> &workloads,
    std::vector<dpu_run_config_t> &run_configs,
    std::vector<dpu_config_padded_t> &configs
) {
    const uint32_t DPU_MRAM_SIZE_BYTES = 62 * 1024 * 1024; // 62MB

    // Fixed MRAM cost: overhead + entire input for one batch item
    const uint32_t mram_overhead = 200
        + context->input.bytes_per_channel * context->input.channels;

    // Marginal MRAM cost per output channel: 2 kernels (shift+sign) + output featuremap
    const uint32_t bytes_per_cout =
        context->output.bytes_per_channel
        + 2 * context->bytes_per_kernel * context->input.channels;

    if (bytes_per_cout == 0 || mram_overhead >= DPU_MRAM_SIZE_BYTES) {
        throw std::runtime_error("build_distribution: MRAM overhead exceeds DPU capacity, this requires tiling of the input which is not currently implemented");
    }
    if (bytes_per_cout > DPU_MRAM_SIZE_BYTES - mram_overhead) {
        throw std::runtime_error("build_distribution: a single output channel does not fit in DPU MRAM");
    }

    // 16 output channels
    // lets say that only 10 fit in a dpu

    // Maximum output channels that fit in one DPU (MRAM-limited)
    const uint32_t max_cout_per_dpu = std::min(
        (uint32_t)context->output.channels,
        (DPU_MRAM_SIZE_BYTES - mram_overhead) / bytes_per_cout
    );

    // max_cout_per_dpu = 10

    // Ceil
    uint32_t dpus_per_batch = (context->output.channels + max_cout_per_dpu - 1) / max_cout_per_dpu;
    // dpus_per_batch = 2
    dpus_per_batch = std::min(dpus_per_batch, (uint32_t)context->output.channels);
    // dpus_per_batch = (2, 16) -> 2

    const uint32_t total_workloads = context->batch_size * dpus_per_batch;
    PRINTF("build_distribution: batch_size=%u, each batch requires %u dpu(s), total_workloads=%u\n",
           context->batch_size, dpus_per_batch, total_workloads);

    workloads.reserve(total_workloads);
    run_configs.reserve(total_workloads);
    configs.reserve(total_workloads);

    const uint32_t base_cout = context->output.channels / dpus_per_batch;
    const uint32_t extra_cout = context->output.channels % dpus_per_batch;

    uint32_t workload_id = 0;

    for (uint32_t batch = 0; batch < context->batch_size; batch++) {
        uint32_t cout_offset = 0;

        for (uint32_t d = 0; d < dpus_per_batch; d++) {
            const uint32_t cout_count = base_cout + (d < extra_cout ? 1u : 0u);

            const uint32_t run_config_idx = run_configs.size();
            run_configs.push_back((dpu_run_config_t){
                .start_input_index_offset = 0,
                .input_index_offset       = 0,
                .input_count              = context->input.channels,
                .start_kernel_index_offset = 0,
                .kernel_index_offset      = 0,
                .kernel_count             = (uint16_t)cout_count,
                // .kernel_count             = context->input.channels
            });

            const uint32_t config_idx = configs.size();
            dpu_config_padded_t cfg = {};
            cfg.config.num_configs = 1;
            cfg.config.num_inputs  = context->input.channels;
            cfg.config.num_outputs = cout_count;
            cfg.config.num_kernels = cout_count;
            cfg.config.num_runs    = 1;
            cfg.config.run_config_base_bytes = 0;

            cfg.config.input_base_bytes = MULTIPLEOF8BYTES(
                sizeof(dpu_run_config_t));

            cfg.config.shiftkernel_base_bytes = MULTIPLEOF8BYTES(
                cfg.config.input_base_bytes
                + context->input.channels * context->input.bytes_per_channel);

            cfg.config.signkernel_base_bytes = MULTIPLEOF8BYTES(
                cfg.config.shiftkernel_base_bytes
                + cout_count * context->input.channels * context->bytes_per_kernel);

            cfg.config.output_base_bytes = MULTIPLEOF8BYTES(
                cfg.config.signkernel_base_bytes
                + cout_count * context->input.channels * context->bytes_per_kernel);

            configs.push_back(cfg);

            workloads.push_back((dpu_workload_t){
                .workload_id      = workload_id++,
                .batch_id         = batch,
                .run_config_count = 1,
                .cout_offset      = cout_offset,
                .cout_count       = cout_count,
                .input_addr       = legacy_find_input_addr(context, batch, 0),
                .output_addr      = legacy_find_output_addr(context, batch, cout_offset),
                .shift_addr       = legacy_find_shift_kernel_addr(context, cout_offset),
                .sign_addr        = legacy_find_sign_kernel_addr(context, cout_offset),
                .run_config_addr  = &run_configs[run_config_idx],
                .config_addr      = &configs[config_idx],
            });

            cout_offset += cout_count;
        }
    }
}

struct host_src_pointers_t {
    const int32_t* input_ptr;
    const int8_t* shift_ptr;
    const int8_t* sign_ptr;
};

// struct GatherContext {
//     struct host_src_pointers_t host_ptrs;
//     const std::vector<dpu_workload_t> &workloads;
//     const std::vector<dpu_config_padded_t>  &configs;
//     const std::vector<dpu_run_config_t> &run_configs;
// };

static bool legacy_configure_block(sg_block_info *out,
                             uint32_t dpu_index,
                             uint32_t block_index,
                             void *args)
{
    char filename[256];
    FILE *file = nullptr;

    GatherContext *ctx = reinterpret_cast<GatherContext *>(args);
    // PRINTF("Configuring DPU %u, block %u: \n", dpu_index, block_index);
    uint32_t workload_index = (*ctx->offset) + dpu_index;
    // PRINTF("Workload index: %u\n", workload_index);
    switch(block_index) {
        // Run configs
        case 0:
            out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].run_config_addr);
            out->length = MULTIPLEOF8BYTES(sizeof(dpu_run_config_t));
            // PRINTF("Configuring DPU %u, block: run configs\n", dpu_index);
            break;
        // Inputs
        case 1:
            out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].input_addr);
            out->length = MULTIPLEOF8BYTES(ctx->context->input.channels * ctx->context->input.bytes_per_channel);
            // PRINTF("Configuring DPU %u, block: inputs\n", dpu_index);
            // // debug print whatever were sending:
            // snPRINTF(filename, sizeof(filename), "inputs_dpu_%u.txt", dpu_index);
            // file = fopen(filename, "w");
            // for (int i = 0; i < out->length; i++) {
            //     int8_t val = ((int8_t*)out->addr)[i];
            //     fPRINTF(file, "%d ", val);
            //     if (i % ctx->context->input.bytes_per_channel == ctx->context->input.bytes_per_channel - 1) {
            //         fPRINTF(file, "\n");
            //     }
            // }
            // fclose(file);
            // PRINTF("\n");
            break;
        // Shifts
        case 2:
            out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].shift_addr);
            out->length = MULTIPLEOF8BYTES(ctx->context->bytes_per_kernel * ctx->workloads[workload_index].cout_count * ctx->context->input.channels);
            // debug print whatever were sending:
            // Write it to a file instead of printing to console, because it can be very large
            // The filename should be prefixed with the DPU index
            // snPRINTF(filename, sizeof(filename), "shifts_output_dpu_%u.txt", dpu_index);
            // file = fopen(filename, "w");
            // for (int i = 0; i < out->length; i++) {
            //     int8_t val = ((int8_t*)out->addr)[i];
            //     fPRINTF(file, "%d ", val);
            //     if (i % ctx->context->bytes_per_kernel == ctx->context->bytes_per_kernel - 1) {
            //         fPRINTF(file, "\n");
            //     }
            // }
            // fclose(file);
            // PRINTF("Configuring DPU %u, block: shifts\n", dpu_index);
            break;
        // Signs
        case 3:
            out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].sign_addr);
            out->length = MULTIPLEOF8BYTES(ctx->context->bytes_per_kernel * ctx->workloads[workload_index].cout_count * ctx->context->input.channels);
            // debug print whatever were sending:
            // snPRINTF(filename, sizeof(filename), "signs_output_dpu_%u.txt", dpu_index);
            // file = fopen(filename, "w");
            // for (int i = 0; i < out->length; i++) {
            //     int8_t val = ((int8_t*)out->addr)[i];
            //     fPRINTF(file, "%d ", val);
            //     if (i % ctx->context->bytes_per_kernel == ctx->context->bytes_per_kernel - 1) {
            //         fPRINTF(file, "\n");
            //     }
            // }
            // fclose(file);
            // PRINTF("Configuring DPU %u, block: signs\n", dpu_index);
            break;
        default:
            // PRINTF("Invalid block index requested: %u\n", block_index);
            return false;
    }
    // PRINTF("Block configured: addr=%p, length=%u\n", out->addr, out->length);

    // return true;
    return true;
}

// void dump_mram(struct dpu_set_t set, uint8_t *buffer, uint8_t *config_buffer, uint32_t buffer_size, uint32_t config_buffer_size) {
//     dpu_set_t dpu;
//     uint32_t dpu_id = 0;

//     DPU_FOREACH(set, dpu, dpu_id) {
//         if (dpu_id != 0) {
//             continue;
//         }
//         PRINTF("Dumping first %u bytes of mram_arena from DPU %u\n", buffer_size, dpu_id);
//         DPU_ASSERT(dpu_copy_from(dpu, "mram_arena", 0, buffer, buffer_size));
//         DPU_ASSERT(dpu_copy_from(dpu, "config", 0, config_buffer, config_buffer_size));
//     }
// }

/**
 * Verifies the input tensors for the DPU kernel
 * @param context The context containing kernel parameters
 * @param input The input tensor
 * @param shift The shift tensor
 * @param sign The sign tensor
 * @return 0 if verification passes, -1 otherwise
 */
static int legacy_verify_input(
    const struct Context *context,
    at::Tensor input,
    at::Tensor shift,
    at::Tensor sign,
    int32_t stride,
    int32_t padding
) {
    STD_TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    STD_TORCH_CHECK(shift.is_contiguous(), "Shift must be contiguous");
    STD_TORCH_CHECK(sign.is_contiguous(),  "Sign must be contiguous");
    STD_TORCH_CHECK(input.scalar_type() == c10::ScalarType::Int,
                    "Input must be int32");
    STD_TORCH_CHECK(shift.scalar_type() == c10::ScalarType::Char,
                    "Shift must be int8");
    STD_TORCH_CHECK(sign.scalar_type()  == c10::ScalarType::Char,
                    "Sign must be int8");
    STD_TORCH_CHECK(stride == 1, "DPU kernel only supports stride=1");

    // STD_TORCH_CHECK(Cin <= DPU_MAX_INPUT_FEATURES,
    //                 "Cin exceeds DPU limit of ", DPU_MAX_INPUT_FEATURES);
    return 0;
}

at::Tensor convolution_kernel(
    DpuPool &dpu_pool,
    at::Tensor input,
    at::Tensor shift,
    at::Tensor sign,
    int32_t stride,
    int32_t padding,
    std::string dpu_binary,
    bool dpu_logs,
    int32_t dpu_limit,
    std::string timing_log_file
) {

    TICK(t_total);
    // Extract dimensions and other parameters that are needed in the program
    Context context = {
        .batch_size = (uint16_t) input.size(0),
        .kernel_size = (uint16_t) shift.size(2),
        .padding = (uint16_t) padding,
        .stride = (uint16_t) stride,

        .input = {
            .channels = (uint16_t) input.size(1),
            .height = (uint16_t) input.size(2),
            .width = (uint16_t) input.size(3),
            .bytes_per_channel = 0,
        },
        .output = {
            .channels = (uint16_t) shift.size(0),
            .height = (uint16_t) (input.size(2) + (2 * padding) - shift.size(2) + 1),
            .width = (uint16_t) (input.size(3) + (2 * padding) - shift.size(2) + 1),
            .bytes_per_channel = 0,
        },
        .input_base_addr = (int32_t*) input.const_data_ptr<int32_t>(),
        .shift_base_addr = (int8_t*) shift.const_data_ptr<int8_t>(),
        .sign_base_addr = (int8_t*) sign.const_data_ptr<int8_t>(),
        .output_base_addr = nullptr,
    };

    // Set sizes
    context.input.bytes_per_channel = context.input.height * context.input.width * sizeof(int32_t);
    context.output.bytes_per_channel = context.output.height * context.output.width * sizeof(int32_t);
    context.bytes_per_kernel = context.kernel_size * context.kernel_size * sizeof(int8_t);

    if (legacy_verify_input(&context, input, shift, sign, stride, padding) != 0) {
        std::cerr << "Input verification failed" << std::endl;
        return at::Tensor();
    }

    TICK(t_tensor_allocation);
    // Create output tensor on the same device as input
    at::Tensor output = at::empty(
        {context.batch_size, context.output.channels, context.output.height, context.output.width},
        c10::ScalarType::Int,
        std::nullopt,                  // layout
        input.device(),                // torch::stable::Device
        std::nullopt,                  // pin_memory
        torch::headeronly::MemoryFormat::Contiguous
    );
    double t_tensor_ms = ms_since(t_tensor_allocation);

    context.output_base_addr = (int32_t*) output.const_data_ptr<int32_t>();


    // Each dpu will receive the entire Cin channels for a subset of the Cout output channels.
    // The DPUs will compute all spatial locations for their assigned output
    // channels before moving on to the next set of output channels.
    // This means we need enough DPUs to cover all Cout channels,
    // even if some DPUs will be underutilized for the last few output channels.
    // const int32_t output_count = (N * Cout);
    const int32_t DPU_LIMIT = dpu_limit;


    std::vector<dpu_workload_t> workloads;
    std::vector<dpu_run_config_t> dpu_run_configs;
    std::vector<dpu_config_padded_t> dpu_configs;

    TICK(t_dist);
    legacy_build_distribution(&context, workloads, dpu_run_configs, dpu_configs);
    double t_dist_ms = ms_since(t_dist);

    for (int64_t i = 0; i < (int64_t) workloads.size(); i++) {
        PRINTF("DPU %lld: N[%u], Cin[0:%lld], Cout[%u:%u]\n",
            (long long)i,
            workloads[i].batch_id,
            (long long)(context.input.channels - 1),
            workloads[i].cout_offset,
            (uint16_t)(workloads[i].cout_offset + workloads[i].cout_count - 1));
    }

    const uint32_t dpus_needed = std::min((uint32_t) workloads.size(), (uint32_t) DPU_LIMIT);
    const uint32_t dpu_batching_count = (workloads.size() + dpus_needed - 1) / dpus_needed;
    PRINTF("Total workloads: %lu, DPUs needed: %u, DPU batching count: %u\n",
           workloads.size(), dpus_needed, dpu_batching_count);

    dpu_set_t dpu;
    uint32_t nr_dpus;

    // Allocate DPUs
    TICK(t_alloc);
    // if (dpus_needed < 5) {
    //     PRINTF("Requesting too few DPUs (%u) for sg transfers, setting \"sgXferMaxBlocksPerDpu=5\"...\n", dpus_needed);
    //     DPU_ASSERT(dpu_alloc(dpus_needed, "sgXferEnable=true,sgXferMaxBlocksPerDpu=5", &set));
    // } else {
    //     DPU_ASSERT(dpu_alloc(dpus_needed, "sgXferEnable=true", &set));
    // }
	// DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    // double t_alloc_ms = ms_since(t_alloc);
    dpu_pool.allocate(dpus_needed);
    PRINTF("Allocated %d/%d DPUs\n", nr_dpus, dpus_needed);

    // Load binary
    TICK(t_load);
    // DPU_ASSERT(dpu_load(set, dpu_binary.c_str(), NULL));
    dpu_pool.load(dpu_binary.c_str());
    double t_load_ms = ms_since(t_load);
    PRINTF("Loaded DPU binary: %s\n", dpu_binary.c_str());

    uint32_t workload_offset = 0;
    GatherContext Gcontext = {
        .context = &context,
        .workloads = workloads,
        .offset = &workload_offset,
    };
    get_block_t get_block_info = {
        .f = legacy_configure_block,
        .args = &Gcontext,
        .args_size = sizeof(Gcontext)
    };

    double t_xfer_config_ms = 0, t_xfer_sg_ms = 0, t_launch_ms = 0, t_gather_ms = 0;

    for (int dpu_batch = 0; dpu_batch < (int64_t) dpu_batching_count; dpu_batch++) {
        PRINTF("Running batch %lld/%u on DPUs...\n", (long long)dpu_batch + 1, dpu_batching_count);

        PRINTF("Sending configs to DPUs\n");
        TICK(t_cfg);
        uint32_t dpu_index = 0;
        DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[workload_offset + dpu_index].config_addr));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_pool.set, DPU_XFER_TO_DPU, "config", 0, sizeof(dpu_config_padded_t), DPU_XFER_DEFAULT));
        t_xfer_config_ms += ms_since(t_cfg);
        PRINTF("Configs sent to DPUs\n");

        PRINTF("Starting scatter-gather transfer to DPUs\n");
        PRINTF("Expected size: %u bytes per DPU\n", dpu_configs[workload_offset].config.output_base_bytes);
        PRINTF("dpu_index: %u, workload_offset: %u\n", dpu_index, workload_offset);
        TICK(t_sg);
        DPU_ASSERT(dpu_push_sg_xfer(dpu_pool.set, DPU_XFER_TO_DPU, "mram_arena", 0,
                                dpu_configs[workload_offset].config.output_base_bytes,
                                &get_block_info, DPU_SG_XFER_DEFAULT));
        t_xfer_sg_ms += ms_since(t_sg);

    //     {
    //     const uint32_t BUFF_SIZE = 20000;
    //     const uint32_t CONFIG_BUFF_SIZE = sizeof(dpu_config_padded_t);
    //     uint8_t *mram_buf = (uint8_t *) malloc(BUFF_SIZE);
    //     uint8_t *config_buf = (uint8_t *) malloc(CONFIG_BUFF_SIZE);

    //     dump_mram(set, mram_buf, config_buf, BUFF_SIZE, CONFIG_BUFF_SIZE);

    //     FILE *hf = fopen("/workspaces/ubuntu-upmem/upmem_menke/dpu/mram_init.h", "w");
    //     if (hf) {
    //         fPRINTF(hf, "#ifndef MRAM_INIT_H\n");
    //         fPRINTF(hf, "#define MRAM_INIT_H\n\n");
    //         fPRINTF(hf, "#include <stdint.h>\n\n");
    //         fPRINTF(hf, "#include \"dpu.h\"\n\n");
    //         fPRINTF(hf, "#include \"mram.h\"\n\n");

    //         // mram_arena dump
    //         fPRINTF(hf, "#define MRAM_INIT_SIZE %u\n\n", BUFF_SIZE);
    //         fPRINTF(hf, "__dma_aligned __mram uint8_t mram_init[MRAM_INIT_SIZE] = {\n");
    //         for (uint32_t i = 0; i < BUFF_SIZE; i++) {
    //             if (i % 16 == 0) fPRINTF(hf, "    ");
    //             fPRINTF(hf, "0x%02x", mram_buf[i]);
    //             if (i + 1 < BUFF_SIZE) fPRINTF(hf, ",");
    //             if (i % 16 == 15) fPRINTF(hf, "\n");
    //             else fPRINTF(hf, " ");
    //         }
    //         fPRINTF(hf, "\n};\n\n");

    //         // config symbol dump
    //         fPRINTF(hf, "#define CONFIG_INIT_SIZE %u\n\n", CONFIG_BUFF_SIZE);
    //         fPRINTF(hf, "__dma_aligned __mram uint8_t config_init[CONFIG_INIT_SIZE] = {\n");
    //         for (uint32_t i = 0; i < CONFIG_BUFF_SIZE; i++) {
    //             if (i % 16 == 0) fPRINTF(hf, "    ");
    //             fPRINTF(hf, "0x%02x", config_buf[i]);
    //             if (i + 1 < CONFIG_BUFF_SIZE) fPRINTF(hf, ",");
    //             if (i % 16 == 15) fPRINTF(hf, "\n");
    //             else fPRINTF(hf, " ");
    //         }
    //         fPRINTF(hf, "\n};\n");
    //         fPRINTF(hf, "\n#endif\n");

    //         fclose(hf);
    //         PRINTF("MRAM dump written to mram_init.h\n");
    //     } else {
    //         PRINTF("Failed to open mram_init.h for writing\n");
    //     }
    // }

        PRINTF("Starting DPUs\n");
        TICK(t_launch);
        DPU_ASSERT(dpu_launch(dpu_pool.set, DPU_SYNCHRONOUS));
        t_launch_ms += ms_since(t_launch);
        PRINTF("DPUs finished execution\n");

        PRINTF("Scattering results from DPUs\n");
        TICK(t_gather);
        dpu_index = 0;
        DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[workload_offset + dpu_index].output_addr));
        }
        uint32_t output_size = workloads[workload_offset].cout_count * context.output.bytes_per_channel;
        uint32_t mram_arena_offset = workloads[workload_offset].config_addr->config.output_base_bytes;
        DPU_ASSERT(dpu_push_xfer(dpu_pool.set, DPU_XFER_FROM_DPU, "mram_arena", mram_arena_offset, output_size, DPU_XFER_DEFAULT));
        t_gather_ms += ms_since(t_gather);
        PRINTF("Results scattered from DPUs\n\n\n");

        workload_offset += nr_dpus;
    }

    double t_total_ms = ms_since(t_total);

    // if (!timing_log_file.empty()) {
    //     static std::set<std::string> s_headers_written;
    //     FILE *tf = fopen(timing_log_file.c_str(), "a");
    //     if (tf) {
    //         if (s_headers_written.find(timing_log_file) == s_headers_written.end()) {
    //             fprintf(tf, "cin,cout,kernel_size,tensor_alloc_ms,build_dist_ms,dpu_alloc_ms,dpu_load_ms,xfer_config_ms,sg_xfer_ms,dpu_launch_ms,xfer_results_ms,total_ms\n");
    //             s_headers_written.insert(timing_log_file);
    //         }
    //         fprintf(tf, "%lld,%lld,%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
    //             (long long)context.input.channels, (long long)context.output.channels, (long long)context.kernel_size,
    //             t_tensor_ms,
    //             t_dist_ms,
    //             t_alloc_ms,
    //             t_load_ms,
    //             t_xfer_config_ms,
    //             t_xfer_sg_ms,
    //             t_launch_ms,
    //             t_gather_ms,
    //             t_total_ms );
    //         fclose(tf);
    //     }
    // }

    // DPU_ASSERT(dpu_free(set));
    return output;
}

// // Register operator schema
// STABLE_TORCH_LIBRARY(dpu, m) {
//     m.def("conv2d(Tensor input, Tensor shift, Tensor sign, int stride, int padding, str dpu_binary, bool dpu_logs, int dpu_limit, str timing_log_file) -> Tensor");
// }

// // Register CPU implementation
// STABLE_TORCH_LIBRARY_IMPL(dpu, CPU, m) {
//     m.impl("conv2d", TORCH_BOX(&convolution_kernel));
// }