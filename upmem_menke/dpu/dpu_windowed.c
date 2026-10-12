#include <attributes.h>
#include <buddy_alloc.h>
#include <mram.h>
#include <stdint.h>
#include <profiling.h>
#include <stdio.h>
#include <alloc.h>
#include <string.h>
#include "dpu.h"
#include <handshake.h>
#include <stdint.h>
#include <defs.h>
#include <barrier.h>
#include "worker.h"
#include "governor.h"
#include "utils.h"


// WRAM
struct window_t windows[NUM_WINDOWS];

struct tasklet_t tasklets[NR_TASKLETS];
struct governor_t governor;

// MRAM
__dma_aligned __mram_noinit struct config_padded_t mconfig;
__dma_aligned struct config_padded_t config;
__dma_aligned __mram uint32_t output_addresses[256];
__dma_aligned __mram uint32_t input_addresses[256];
#if RESTORE_MRAM
#include "mram_init.h"
#else
__dma_aligned __mram_noinit int8_t mram_arena[ARENA_SIZE_BYTES];
#endif

BARRIER_INIT(barrier, NR_TASKLETS);

PROFILING_INIT(governor_boot);
PROFILING_INIT(worker_boot);

PROFILING_INIT(worker_wait_before_work);
PROFILING_INIT(worker_process_window);

int main() {
    uint8_t tasklet_id = me();

    DPU_PRINTF("DPU tasklet %d/%d started\n", tasklet_id + 1, NR_TASKLETS);
    DPU_PRINTF("DPU tasklet %d stack size: %i\n", tasklet_id + 1, check_stack());

    if (tasklet_id == 0) {
        profiling_start(&governor_boot);

        #if RESTORE_MRAM

        // // Load mram_arena from the host-dumped snapshot
        // // We can only copy 2048 bytes at a time, so we need to do it in chunks
        // int32_t bytes_left = MRAM_INIT_SIZE;
        // printf("MRAM INIT SIZE: %u\n", MRAM_INIT_SIZE);
        buddy_init(2048);
        uint8_t *buffer = (uint8_t *) buddy_alloc(2048);
        // while (bytes_left > 0) {
        //     uint16_t size = (bytes_left > 2048) ? 2048 : bytes_left;
        //     uint32_t offset = MRAM_INIT_SIZE - bytes_left;
        //     printf("Restoring MRAM[%u] with %u bytes\n", offset, size);
        //     mram_read(&mram_init[offset], buffer, size);
        //     mram_write(buffer, &mram_arena[offset], size);
        //     printf("Bytes done: %u/%u\n", bytes_left, size);
        //     bytes_left = bytes_left - size;
        // }

        mram_read(&config_init, buffer, CONFIG_INIT_SIZE);
        mram_write(buffer, &mconfig, CONFIG_INIT_SIZE);

        buddy_free(buffer);

        #endif

        

        // mconfig.config.output_base_bytes = mconfig.config.input_base_bytes;
        governor_initialize(&governor);
        governor_work(&governor);
    // tasklets 1-N are workers
    } else {
        profiling_start(&worker_boot);
        DPU_PRINTF("Worker[%d], tasklet %d initialized, waiting for governor to release\n", tasklet_id - 1, tasklet_id);
        worker_initialize(&tasklets[tasklet_id - 1], tasklet_id - 1);
        worker_work(&tasklets[tasklet_id - 1], &governor);
    }
}
