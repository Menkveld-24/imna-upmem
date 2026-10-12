#ifndef GOVERNOR_H
#define GOVERNOR_H

#include "dpu.h"
#include <barrier.h>
#include <profiling_internals.h>
#include "mram.h"

extern struct tasklet_t tasklets[NR_TASKLETS];
extern struct governor_t governor;

extern barrier_t barrier;
extern dpu_profiling_t governor_boot;

extern __dma_aligned __mram_noinit struct config_padded_t mconfig;
extern __dma_aligned struct config_padded_t config;
// extern __dma_aligned __mram_noinit struct run_config_t mconfig;

// extern __dma_aligned __mram int8_t m_shiftkernels[MAX_KERNEL_COUNT][KERNEL_SIZE * KERNEL_SIZE];
// extern __dma_aligned __mram int8_t m_signkernels[MAX_KERNEL_COUNT][KERNEL_SIZE * KERNEL_SIZE];
// extern __dma_aligned __mram fxp32_t input_buffer[MAX_INPUT_FEATURES][INPUT_FEATURE_SIZE];
// extern __dma_aligned __mram fxp32_t output_buffer[MAX_OUTPUT_FEATURES][OUTPUT_FEATURE_SIZE];
#if RESTORE_MRAM
extern __dma_aligned __mram int8_t mram_arena[ARENA_SIZE_BYTES];
#else 
extern __dma_aligned __mram_noinit int8_t mram_arena[ARENA_SIZE_BYTES];
#endif
extern struct window_t windows[NUM_WINDOWS];

void governor_initialize(struct governor_t *governor);
void governor_work(struct governor_t *governor);

#endif
