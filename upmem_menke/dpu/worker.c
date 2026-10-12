#include "dpu.h"
#include <defs.h>
#include <attributes.h>
#include <mram.h>
#include <profiling.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "governor.h"
#include "mram.h"
#include <barrier.h>
#include "worker.h"
#include "utils.h"
#include "built_ins.h"

// Define the supported convolution configurations
#define K_SIZE 3
#define K_STRIDE 1
#include "./layers/conv2d.h"

#define K_SIZE 7
#define K_STRIDE 1
#include "./layers/conv2d.h"

// Define the supported ReLU configuration
#include "./layers/relu.h"

// Define the supported Max Pooling configuration
#define K_SIZE 2
#define K_STRIDE 2
#include "./layers/maxpool.h"

/**
 * Initialize the worker tasklet
 */
void worker_initialize(struct tasklet_t *tasklet, uint8_t tasklet_id) {
    tasklet->id = tasklet_id;
    tasklet->working_window = (struct window_position_t){
        .id = 1,
        .window_index = 0
    };
    tasklet->last_processed_window = (struct window_position_t){0, 0};

    profiling_stop(&worker_boot);
    DPU_PRINTF("Worker %d initialized\n", tasklet_id);
    // profiling_start(&worker_wait_before_work);
    // Wait for the governor to release us
    barrier_wait(&barrier);
}

/**
 * Marks the current working window as processed and updates the tasklet's
 * last processed window position.
 */
void markWindowProcessed(struct tasklet_t *tasklet) {
    incrementWindowPosition(&tasklet->last_processed_window);
    incrementWindowPosition(&tasklet->working_window);
}

static void convolution(
    const struct tasklet_t *tasklet,
    struct window_t *working_window,
    uint8_t opcode
) {
    uint32_t kernel_size = config.config.kernel_size;
    uint32_t stride = config.config.stride;
    if (kernel_size == 3 && stride == 1) {
        conv2d_3_1_doConvolutionForWindow(
            tasklet,
            working_window
        );
    } else if (kernel_size == 7 && stride == 1) {
        conv2d_7_1_doConvolutionForWindow(
            tasklet,
            working_window
        );
    } else {
        DPU_PRINTF("Unsupported convolution config: kernel size %d, stride %d\n", kernel_size, stride);
        // Exit prematurely
        return;
    }

    if (working_window->header.last_window_before_saving && shouldPerformOP(OP_RELU, opcode)) {
        doReluForWindowAfterConv2D(
            tasklet,
            working_window
        );
    }
}

static void maxpooling(
    const struct tasklet_t *tasklet,
    struct window_t *working_window,
    uint8_t opcode
) {
    uint32_t kernel_size = config.config.kernel_size;
    uint32_t stride = config.config.stride;
    if (kernel_size == 2 && stride == 2) {
        maxpool_2_2_doMaxPoolingForWindow(
            tasklet,
            working_window
        );
    } else {
        DPU_PRINTF("Unsupported maxpooling config: kernel size %d, stride %d\n", kernel_size, stride);
        // Exit prematurely
        return;
    }
}

/**
 * Main worker function that processes windows assigned by the governor.
 */
void worker_work(struct tasklet_t *tasklet, struct governor_t *governor) {
    DPU_PRINTF("[Worker %d] started working\n", tasklet->id);
    // profiling_stop(&worker_wait_before_work);
    // As long as there's work to do, keep working
    while (governor->total_windows_allocated.id > tasklet->last_processed_window.id) {
        DPU_PRINTF("[Worker %d] Starting work on window[%d], id: %d\n", tasklet->id, tasklet->working_window.window_index, tasklet->working_window.id);
        profiling_start(&worker_process_window);
        // Wait for more work
        bool waiting = false;
        while (tasklet->working_window.id > governor->latest_prepared_window.id) {
            DPU_PRINTF("[Worker %d] waiting for work (last free window: %d)\n", tasklet->id, governor->latest_prepared_window.id);
            if (!waiting) {
                profiling_start(&worker_wait_before_work);
                waiting = true;
            }
            __builtin_nop_();
        }

        if (waiting) {
            profiling_stop(&worker_wait_before_work);
        }

        struct window_t *current_window = &windows[tasklet->working_window.window_index];
        uint8_t opcode = config.config.opcode;

        DPU_PRINTF("[Worker %d] Starting to process window id: %d with opcode: %02x\n", tasklet->id, current_window->header.id, opcode);
        if (shouldPerformOP(OP_CONV2D, opcode)) {
            DPU_PRINTF("[Worker %d] Performing convolution for window id: %d\n", tasklet->id, current_window->header.id);
            convolution(tasklet, current_window, opcode);
        } else if (shouldPerformOP(OP_RELU, opcode)) {
            DPU_PRINTF("ReLU operation is currently not supported on the DPU\n");
            return;
        } else if (shouldPerformOP(OP_MAXPOOL, opcode)) {
            maxpooling(tasklet, current_window, opcode);
        } else {
            DPU_PRINTF("Unsupported opcode: %d\n", opcode);
            return;
        }

        // Mark this window as processed
        DPU_PRINTF("[Worker %d] finished working on window id: %d\n", tasklet->id, tasklet->last_processed_window.id + 1);
        markWindowProcessed(tasklet);
        profiling_stop(&worker_process_window);
    }
    DPU_PRINTF("[Worker %d] (thread %d) No more work to do, exiting\n", tasklet->id, me());
}