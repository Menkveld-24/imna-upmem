#include "sg_xfer.h"
#include <cstdlib>
#include <cstdio>
#include "dpu_dtypes.h"
#include <torch/library.h>

// Relative to the working directory (upmem_ai/); override with -DMRAM_INIT_PATH=...
#ifndef MRAM_INIT_PATH
#define MRAM_INIT_PATH "../upmem_menke/dpu/mram_init.h"
#endif

// bool configure_block(sg_block_info *out,
//                      uint32_t dpu_index,
//                      uint32_t block_index,
//                      void *args)
// {
//     char filename[256];
//     FILE *file = nullptr;

//     GatherContext *ctx = reinterpret_cast<GatherContext *>(args);
//     // PRINTF("Configuring DPU %u, block %u: \n", dpu_index, block_index);
//     uint32_t workload_index = (*ctx->offset) + dpu_index;
//     // PRINTF("Workload index: %u\n", workload_index);
//     switch(block_index) {
//         // Run configs
//         case 0:
//             out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].run_config_addr);
//             out->length = MULTIPLEOF8BYTES(sizeof(dpu_run_config_t));
//             // PRINTF("Configuring DPU %u, block: run configs\n", dpu_index);
//             break;
//         // Shifts
//         case 1:
//             out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].shift_addr);
//             out->length = MULTIPLEOF8BYTES(ctx->context->bytes_per_kernel * ctx->workloads[workload_index].cout_count * ctx->context->input.channels);
//             break;
//         // Signs
//         case 2:
//             out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].sign_addr);
//             out->length = MULTIPLEOF8BYTES(ctx->context->bytes_per_kernel * ctx->workloads[workload_index].cout_count * ctx->context->input.channels);
//             break;
//         // Inputs
//         case 3:
//             // If the input address is not set, they're already on the dpus
//             if (ctx->workloads[workload_index].input_addr == nullptr) {
//                 return false;
//             }
//             out->addr = reinterpret_cast<uint8_t *>(ctx->workloads[workload_index].input_addr);
//             out->length = MULTIPLEOF8BYTES(ctx->context->input.channels * ctx->context->input.bytes_per_channel);
//             // PRINTF("Configuring DPU %u, block: inputs\n", dpu_index);
//             break;
//         default:
//             // PRINTF("Invalid block index requested: %u\n", block_index);
//             return false;
//     }
//     // PRINTF("Block configured: addr=%p, length=%u\n", out->addr, out->length);

//     return true;
// }

void dump_mram(struct dpu_set_t set, uint8_t *buffer, uint8_t *config_buffer, uint32_t buffer_size, uint32_t config_buffer_size) {
    dpu_set_t dpu;
    uint32_t dpu_id = 0;

    DPU_FOREACH(set, dpu, dpu_id) {
        if (dpu_id != 0) {
            continue;
        }
        PRINTF("Dumping first %u bytes of mram_arena from DPU %u\n", buffer_size, dpu_id);
        DPU_ASSERT(dpu_copy_from(dpu, "mram_arena", 0, buffer, buffer_size));
        DPU_ASSERT(dpu_copy_from(dpu, "mconfig", 0, config_buffer, config_buffer_size));
    }
}

void dump_mram_to_header(struct dpu_set_t dpu_set, uint64_t bytes) {
    // const uint32_t BUFF_SIZE = 20000;
    const uint32_t CONFIG_BUFF_SIZE = sizeof(dpu_config_padded_t);
    printf("Config size: %u bytes\n", CONFIG_BUFF_SIZE);
    uint8_t *mram_buf = (uint8_t *) malloc(bytes);
    uint8_t *config_buf = (uint8_t *) malloc(CONFIG_BUFF_SIZE);

    dump_mram(dpu_set, mram_buf, config_buf, bytes, CONFIG_BUFF_SIZE);
    PRINTF("MRAM dump completed, writing to header file...\n");

    FILE *hf = fopen(MRAM_INIT_PATH, "w");
    if (hf) {
        PRINTF("Opened mram_init.h for writing\n");
        fprintf(hf, "#ifndef MRAM_INIT_H\n");
        fprintf(hf, "#define MRAM_INIT_H\n\n");
        fprintf(hf, "#include <stdint.h>\n\n");
        fprintf(hf, "#include \"dpu.h\"\n\n");
        fprintf(hf, "#include \"mram.h\"\n\n");

        // mram_arena dump
        fprintf(hf, "#define MRAM_INIT_SIZE %u\n\n", bytes);
        fprintf(hf, "__dma_aligned __mram int8_t mram_arena[MRAM_INIT_SIZE] = {\n");
        for (uint32_t i = 0; i < bytes; i++) {
            if (i % 16 == 0) fprintf(hf, "    ");
            fprintf(hf, "0x%02x", mram_buf[i]);
            if (i + 1 < bytes) fprintf(hf, ",");
            if (i % 16 == 15) fprintf(hf, "\n");
            else fprintf(hf, " ");
        }
        fprintf(hf, "\n};\n\n");

        // config symbol dump
        fprintf(hf, "#define CONFIG_INIT_SIZE %u\n\n", CONFIG_BUFF_SIZE);
        fprintf(hf, "__dma_aligned __mram uint8_t config_init[CONFIG_INIT_SIZE] = {\n");
        for (uint32_t i = 0; i < CONFIG_BUFF_SIZE; i++) {
            if (i % 16 == 0) fprintf(hf, "    ");
            fprintf(hf, "0x%02x", config_buf[i]);
            if (i + 1 < CONFIG_BUFF_SIZE) fprintf(hf, ",");
            if (i % 16 == 15) fprintf(hf, "\n");
            else fprintf(hf, " ");
        }
        fprintf(hf, "\n};\n");
        fprintf(hf, "\n#endif\n");

        fclose(hf);
        PRINTF("MRAM dump written to mram_init.h\n");
    } else {
        PRINTF("Failed to open mram_init.h for writing\n");
    }

    free(mram_buf);
    free(config_buf);

    TORCH_CHECK(false, "Stopping execution after dumping MRAM for debugging");
}
