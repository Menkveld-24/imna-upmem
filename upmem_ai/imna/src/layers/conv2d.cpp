#include <ATen/ops/empty_like.h>
#include <cstdlib>
#include <torch/library.h>
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
#include <unistd.h>
#include <cerrno>
#include <future>
#include <chrono>
#include <stdexcept>

#include "../csv_timing.h"
#include "../sg_xfer.h"
#include "conv2d.h"
#include "../timing.h"
#include "dpu.h"

static int verify_input(
    const struct Context *context,
    DpuTensor input,
    at::Tensor fused_ss,
    int64_t stride,
    int64_t padding
) {
    if (input.onCPU()) {
        TORCH_CHECK(input.data.is_contiguous(), "Input must be contiguous");
    }
    TORCH_CHECK(fused_ss.is_contiguous(), "Shift+sign must be contiguous");
    // TORCH_CHECK(input.dtype() == at::kInt,  "Input must be int32");
    TORCH_CHECK(fused_ss.scalar_type() == at::kChar, "Fused shift+sign must be int8");
    TORCH_CHECK(stride == 1, "DPU kernel only supports stride=1");
    return 0;
}

DpuTensor eager_conv2d_impl(
    at::Tensor input,
    at::Tensor fused_ss,
    int64_t stride,
    int64_t padding,
    bool applyReluAfter,
    DpuPool& dpu_pool
) {
    // Extract dimensions and other parameters that are needed in the program
    Context context = {
        .batch_size = (uint16_t) input.size(0),
        .kernel_size = (uint16_t) fused_ss.size(2),
        .padding = (uint16_t) padding,
        .stride = (uint16_t) stride,

        .input = {
            .channels = (uint16_t) input.size(1),
            .height = (uint16_t) input.size(2),
            .width = (uint16_t) input.size(3),
            .bytes_per_channel = 0,
        },
        .output = {
            .channels = (uint16_t) fused_ss.size(0),
            .height = (uint16_t) (input.size(2) + (2 * padding) - fused_ss.size(2) + 1),
            .width = (uint16_t) (input.size(3) + (2 * padding) - fused_ss.size(2) + 1),
            .bytes_per_channel = 0,
        },
        .input_base_addr = (int32_t*) input.data_ptr<int32_t>(),
        .fused_ss_base_addr = (int8_t*) fused_ss.data_ptr<int8_t>(),
        .output_base_addr = nullptr,
    };

    // Set sizes
    context.input.bytes_per_channel = context.input.height * context.input.width * sizeof(int32_t);
    context.output.bytes_per_channel = context.output.height * context.output.width * sizeof(int32_t);
    context.bytes_per_kernel = context.kernel_size * context.kernel_size * sizeof(int8_t);
    
    // Do memory partitioning for this convolution
    uint32_t input_offset_bytes = MULTIPLEOF8BYTES(sizeof(dpu_run_config_t));
    uint32_t weight_offset_bytes = input_offset_bytes + MULTIPLEOF8BYTES(context.input.channels * context.input.bytes_per_channel);
    uint32_t output_offset_bytes = weight_offset_bytes + (MULTIPLEOF8BYTES(context.bytes_per_kernel * context.input.channels * context.output.channels) * 2);

    MramPartitioning partitioning = {
        .input_offset_bytes  = (uint32_t) (input_offset_bytes),
        .weight_offset_bytes  = (uint32_t) (weight_offset_bytes),
        .output_offset_bytes   = (uint32_t) (output_offset_bytes),
    };
    DpuTensor i = DpuTensor(input.contiguous());
    
    return mram_conv2d_impl(
        i,
        fused_ss,
        stride,
        padding,
        applyReluAfter,
        &partitioning,
        dpu_pool
    );
}

// ---------------------------------------------------------------------------
// Core implementation — takes an explicit DpuPool so it can be shared with
// the UpmemAccel class in accel_class.cpp.
// ---------------------------------------------------------------------------

DpuTensor mram_conv2d_impl(
    DpuTensor input,
    at::Tensor fused_ss,
    int64_t stride,
    int64_t padding,
    bool applyReluAfter,
    const struct MramPartitioning *mram_partitioning,
    DpuPool& dpu_pool
) {
    Context context = {
        .batch_size  = (uint16_t) input.data_sizes[0],
        .kernel_size = (uint16_t) fused_ss.size(2),
        .padding     = (uint16_t) padding,
        .stride      = (uint16_t) stride,

        .input = {
            .channels          = (uint16_t) input.data_sizes[1],
            .height            = (uint16_t) input.data_sizes[2],
            .width             = (uint16_t) input.data_sizes[3],
            .bytes_per_channel = 0,
        },
        .output = {
            .channels          = (uint16_t) fused_ss.size(0),
            .height            = (uint16_t) (input.data_sizes[2] + (2 * padding) - fused_ss.size(2) + 1),
            .width             = (uint16_t) (input.data_sizes[3] + (2 * padding) - fused_ss.size(2) + 1),
            .bytes_per_channel = 0,
        },
        .input_base_addr  = nullptr,
        .fused_ss_base_addr  = (int8_t*)  fused_ss.const_data_ptr<int8_t>(),
        .output_base_addr = nullptr,
    };


    printf("Input tensor sizes: batch: %d, channels: %d, height: %d, width: %d\n",
        context.batch_size,
        context.input.channels,
        context.input.height,
        context.input.width
    );
    context.input.bytes_per_channel  = context.input.height  * context.input.width  * sizeof(int32_t);
    context.output.bytes_per_channel = context.output.height * context.output.width * sizeof(int32_t);
    context.bytes_per_kernel = context.kernel_size * context.kernel_size * sizeof(int8_t);


    if (verify_input(&context, input, fused_ss, stride, padding) != 0) {
        std::cerr << "Input verification failed" << std::endl;
        return {};
    }

    DpuTensor output;
    output.data_sizes = {
        (int64_t) context.batch_size,
        (int64_t) context.output.channels,
        (int64_t) context.output.height,
        (int64_t) context.output.width
    };
    output.uploadedToDpu((int64_t) mram_partitioning->output_offset_bytes);

    std::vector<dpu_workload_t>         workloads;
    std::vector<dpu_run_config_t>       dpu_run_configs;
    std::vector<dpu_config_padded_t>    dpu_configs;

    uint8_t opcode = OP_CONV2D;
    if (applyReluAfter) {
        opcode |= OP_RELU;
    }

    build_distribution(
        &context,
        &input,
        mram_partitioning,
        workloads,
        dpu_run_configs,
        dpu_configs,
        opcode,
        padding,
        context.kernel_size,
        stride);

    for (int64_t i = 0; i < (int64_t) workloads.size(); i++) {
        PRINTF("DPU %lld: N[%u], Cin[0:%lld], Cout[%u:%u]\n",
            (long long)i,
            workloads[i].batch_id,
            (long long)(context.input.channels - 1),
            workloads[i].cout_offset,
            (uint16_t)(workloads[i].cout_offset + workloads[i].cout_count - 1));
    }

    dpu_set_t dpu;

    dpu_pool.readyCheck();

    // Transfer the configs
    uint32_t dpu_index = 0;
    DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[dpu_index].config_addr));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_pool.set, DPU_XFER_TO_DPU, "mconfig", 0,
                                sizeof(dpu_config_padded_t), DPU_XFER_DEFAULT));

    // Transfer the run configs
    dpu_index = 0;
    PRINTF("Transferring run configs to DPUs...\n");
    DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[dpu_index].run_config_addr));
    }
    DPU_ASSERT(dpu_push_xfer(
            dpu_pool.set,
            DPU_XFER_TO_DPU,
            "mram_arena",
            workloads[0].config_addr->config.run_config_base_bytes,
            MULTIPLEOF8BYTES(sizeof(dpu_run_config_t) * workloads[0].run_config_count),
            DPU_XFER_DEFAULT));

    uint32_t output_addresses[256] = {0};
    for (int64_t i = 0; i < context.output.channels; i++) {
        output_addresses[i] = mram_partitioning->output_offset_bytes + (i * context.output.bytes_per_channel);
    }

    dpu_broadcast_to(
        dpu_pool.set,
        "output_addresses",
        0,
        &output_addresses,
        sizeof(output_addresses),
        DPU_XFER_DEFAULT);

    uint32_t input_addresses[256] = {0};
    for (int64_t i = 0; i < context.input.channels; i++) {
        input_addresses[i] = mram_partitioning->input_offset_bytes + (i * context.input.bytes_per_channel);
    }

    dpu_broadcast_to(
        dpu_pool.set,
        "input_addresses",
        0,
        &input_addresses,
        sizeof(input_addresses),
        DPU_XFER_DEFAULT);

    // Transfer inputs
    if (input.onCPU()) {
        PRINTF("Transferring input tensor to DPUs...\n");
        PRINTF("Input address on CPU: %p\n", (void*) workloads[0].input_addr);
        dpu_index = 0;
        DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[dpu_index].input_addr));
        }
        DPU_ASSERT(dpu_push_xfer(
                dpu_pool.set,
                DPU_XFER_TO_DPU,
                "mram_arena",
                workloads[0].config_addr->config.input_base_bytes,
                MULTIPLEOF8BYTES(context.input.channels * context.input.bytes_per_channel),
                DPU_XFER_DEFAULT));

        input.uploadedToDpu((int64_t) workloads[0].config_addr->config.input_base_bytes);
    }
    // dump_mram_to_header(dpu_pool.set, 20000);

    // Transfer weights
    // ToDo: These can be broadcasted iirc
    dpu_index = 0;
    PRINTF("Transferring shift and sign kernels to DPUs...\n");
    // DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
    //     DPU_ASSERT(dpu_prepare_xfer(dpu, workloads[dpu_index].fused_ss_addr));
    // }
    // DPU_ASSERT(dpu_push_xfer(
    //         dpu_pool.set,
    //         DPU_XFER_TO_DPU,
    //         "mram_arena",
    //         workloads[0].config_addr->config.fused_ss_kernel_base_bytes,
    //         MULTIPLEOF8BYTES(context.bytes_per_kernel * context.input.channels * context.output.channels),
    //         DPU_XFER_DEFAULT));
    
    dpu_broadcast_to(
        dpu_pool.set,
        "mram_arena",
        workloads[0].config_addr->config.fused_ss_kernel_base_bytes,
        workloads[dpu_index].fused_ss_addr,
        MULTIPLEOF8BYTES(context.bytes_per_kernel * context.input.channels * context.output.channels),
        DPU_XFER_DEFAULT);


    PRINTF("Launching DPU execution\n");
    
    DPU_ASSERT(dpu_launch(dpu_pool.set, DPU_SYNCHRONOUS));
    
    // Logs
    // dpu_index = 0;
    // DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
    //     if (dpu_index == 0) {
    //         // Create a file
    //         FILE *log_file = nullptr;
    //         char filename[256];
    //         snprintf(filename, sizeof(filename), "dpu_log.log");
    //         log_file = fopen(filename, "w");
    //         if (log_file == nullptr) {
    //             PRINTF("Failed to open log file %s: %s\n", filename, strerror(errno));
    //             // If we can't open the file, we still want to print logs to stdout, so we don't return here
    //         } else {
    //             PRINTF("Writing DPU logs to file: %s\n", filename);
    //             DPU_ASSERT(dpu_log_read(dpu, log_file));
    //             fclose(log_file);
    //         }
    //     }
    // }

    // if (!timing_csv_file.empty() && batch_id >= 0) {
    //     write_timing_to_csv(
    //         timing_csv_file.c_str(), batch_id, &context,
    //         t_tensor_ms, t_dist_ms, t_alloc_ms, t_load_ms,
    //         t_xfer_config_ms, t_xfer_sg_ms, t_launch_ms, t_gather_ms,
    //         t_total_ms
    //     );
    // }

    return output;
}