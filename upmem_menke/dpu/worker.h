#ifndef WORKER_H
#define WORKER_H

#include "dpu.h"
#include <barrier.h>
#include <profiling_internals.h>

extern barrier_t barrier;
extern dpu_profiling_t worker_boot;
extern dpu_profiling_t worker_wait_before_work;
extern dpu_profiling_t worker_process_window;

extern __dma_aligned __mram_noinit struct config_padded_t mconfig;
extern __dma_aligned struct config_padded_t config;

// extern __dma_aligned __mram_noinit fxp32_t input_buffer[MAX_INPUT_FEATURES][INPUT_FEATURE_SIZE];
// extern __dma_aligned __mram_noinit fxp32_t output_buffer[MAX_OUTPUT_FEATURES][OUTPUT_FEATURE_SIZE];

void worker_initialize(struct tasklet_t *tasklet, uint8_t tasklet_id);
void worker_work(struct tasklet_t *tasklet, struct governor_t *governor);

#endif