#ifndef UTILS_H
#define UTILS_H

#include "dpu.h"
#include "mram.h"

#if RESTORE_MRAM
extern __dma_aligned __mram int8_t mram_arena[ARENA_SIZE_BYTES];
#else 
extern __dma_aligned __mram_noinit int8_t mram_arena[ARENA_SIZE_BYTES];
#endif

extern __dma_aligned __mram uint32_t output_addresses[256];
extern __dma_aligned __mram uint32_t input_addresses[256];
extern __dma_aligned __mram_noinit struct config_padded_t mconfig;
extern __dma_aligned struct config_padded_t config;

#define OP_CONV2D   0b00000001
#define OP_RELU     0b00000010
#define OP_MAXPOOL  0b00000100

uint8_t nextWindowIndex(uint8_t current_index);
uint8_t previousWindowIndex(uint8_t current_index);

void incrementWindowPosition(struct window_position_t *position);
void DMATransfer_MRAM_to_WRAM (
    __mram_ptr void const *mram_src,
    void *wram_dst,
    uint16_t size_bytes
);
void DMATransfer_WRAM_to_MRAM (
    void const *wram_src,
    __mram_ptr void *mram_dst,
    uint16_t size_bytes
);

static inline bool shouldPerformOP(uint8_t operation, uint8_t opcode) {
    return (opcode & operation) != 0;
}

static inline uint8_t modulo8(uint32_t value) {
    return value & 7;
}

static inline __mram_ptr struct run_config_t *get_run_configs_ptr(uint16_t input_idx) {
    return (__mram_ptr struct run_config_t *)&mram_arena[
        config.config.run_config_base_bytes + input_idx * sizeof(struct run_config_t)
    ];
}

static inline __mram_ptr fxp32_t *get_input_ptr(uint16_t input_idx) {
    return (__mram_ptr fxp32_t *) ((uint32_t) &mram_arena + input_addresses[input_idx]);
    // return (__mram_ptr fxp32_t *)&mram_arena[
    //     config.config.input_base_bytes + input_idx * (config.config.input_feature_height * config.config.input_feature_width * sizeof(fxp32_t))
    // ];
}

static inline __mram_ptr fxp32_t *get_output_ptr(uint16_t output_idx) {
    return (__mram_ptr fxp32_t *) ((uint32_t) &mram_arena + output_addresses[output_idx]);
    // return (__mram_ptr fxp32_t *)&mram_arena[
    //     config.config.output_base_bytes + output_idx * (config.config.output_feature_height * config.config.output_feature_width * sizeof(fxp32_t))
    // ];
}

static inline __mram_ptr int8_t *get_fused_ss_kernel_ptr(uint16_t kernel_idx) {
    return (__mram_ptr int8_t *)&mram_arena[
        config.config.fused_ss_kernel_base_bytes + kernel_idx * (config.config.kernel_size * config.config.kernel_size * sizeof(int8_t))
    ];
}

#endif
