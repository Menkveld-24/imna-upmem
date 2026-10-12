#include "dpu.h"
#include "governor.h"
#include "worker.h"
#include <profiling.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "utils.h"
#include "string.h"


/**
 * Copy the run config for a given window
 * This allows the window to make local edits to a run config without affecting the global config in MRAM until its ready to move to the next config
 */
static void copy_run_config_to_window(struct window_t *window, uint8_t run_config_index) {
    DMATransfer_MRAM_to_WRAM(
        get_run_configs_ptr(run_config_index),
        &window->run_config,
        sizeof(struct run_config_padded_t)
    );
}

/**
 * Set up the first tile in a window
 */
static void init_tile(struct window_t *window, uint16_t window_id, uint8_t run_config_index) {
    window->header = (struct window_header_t){
        .id = window_id,
        .free = false,
        .payload_start = 0,
        .payload_size = 0,
        .output_feature_index = 0,
        .tile_info = (struct tile_t) {
            .id = 1,
            .input_horizontal_offset = 0,
            .input_vertical_offset = 0,
            .input_vertical_offset_index = 0,
            .output_horizontal_offset = 0,
            .output_vertical_offset = 0,
            .width = TILE_WIDTH, // Exclude alignment compensation, this allows us to 'safely' overflow the window payload
            .height = TILE_HEIGHT,
            .output_height = 0,
            .output_width = 0,
            .processed = (struct processed_tile_stats_t) {
                .loaded_input_rows = 0,
                .loaded_input_cols = 0,
            }
        }
    };

    if (config.config.kernel_size > 0) {
        window->header.tile_info.output_height = (TILE_HEIGHT - config.config.kernel_size + 1);
        window->header.tile_info.output_width = (TILE_WIDTH - config.config.kernel_size + 1);
    }

    copy_run_config_to_window(window, run_config_index);
}

/**
 * Prepare the next tile in a window
 */
static void prep_next_window(struct window_t *window, struct window_t *next_window) {
    // Always move to a new window id
    next_window->header.id = window->header.id + 1;

    // Total kernels to do could be 512, but index goes from 0-511.
    // next_window->kernel_tracker.kernels_for_window = MIN(
    //     next_window->kernel_tracker.total_kernels_to_do - next_window->kernel_tracker.next_kernel_index,
    //     KERNELS_PER_WINDOW
    // );
    // next_window->progress_tracker.next_input_index += 1;

    // This is done in the load_kernels
    // next_window->kernel_tracker.next_kernel_index += next_window->kernel_tracker.kernels_for_window;

    // DPU_PRINTF("Prepared next window for window id %d with kernel count %d\n",
    //     next_window->header.id,
    //     next_window->kernel_tracker.kernels_for_window
    // );
}

/**
 * Check if there are still inputs left to process for a given window
 */
static bool has_inputs_left_to_process(struct window_t *window) {
    // We either havent processed the entire feature
    return (window->run_config.config.input_index_offset < window->run_config.config.inputs_per_output);

}

/**
 * Prepare the next tile in a window
 * 0 = continue to next tile
 * 1 = move to next output feature
 * 2 = move to next run config
 */
static int prep_next_tile(struct window_t *window, struct window_t *next_window) {
    // Copy the previous tile info
    next_window->header = (struct window_header_t){
        .id = next_window->header.id,
        .free = false,
        .output_feature_index = window->header.output_feature_index,
        .tile_info = (struct tile_t) {
            // Next id
            .id = window->header.tile_info.id + 1,
            // Copy over offsets
            .input_horizontal_offset = window->header.tile_info.input_horizontal_offset,
            .input_vertical_offset = window->header.tile_info.input_vertical_offset,
            .input_vertical_offset_index = window->header.tile_info.input_vertical_offset_index,
            .output_horizontal_offset = window->header.tile_info.output_horizontal_offset,
            .output_vertical_offset = window->header.tile_info.output_vertical_offset,

            // Copy over the widths and heights
            .width = window->header.tile_info.width,
            .height = window->header.tile_info.height,
            // Copy over output sizes
            .output_height = window->header.tile_info.output_height,
            .output_width = window->header.tile_info.output_width,

            // Reset the processed
            .processed = (struct processed_tile_stats_t) {
                .loaded_input_rows = 0,
                .loaded_input_cols = 0,
            }
        }
    };

    // Copy the run config and progress tracker
    next_window->run_config = window->run_config;
        
    // Add the horizontal offset
    uint8_t kernel_size = config.config.kernel_size;
    // uint8_t opcode = config.config.opcode;

    // if (shouldPerformOP(OP_MAXPOOL, opcode)) {
    //     kernel_size = 1;
    // }
    // - kernel size + 1 because of the convolution output size reduction
    next_window->header.tile_info.input_horizontal_offset += (window->header.tile_info.processed.loaded_input_cols - kernel_size + 1);
    // Update the output horizontal offset as well
    next_window->header.tile_info.output_horizontal_offset += window->header.tile_info.output_width;

    DPU_PRINTF("Offsets for window id %d: input_horizontal_offset: %d, input_vertical_offset: %d\n",
        next_window->header.id,
        next_window->header.tile_info.input_horizontal_offset,
        next_window->header.tile_info.input_vertical_offset
    );

    // this entire row has been processed, move to the next rows
    // Moving to a new tile is needed
    if (next_window->header.tile_info.input_horizontal_offset >= config.config.input_feature_width - kernel_size + 1) {
        DPU_PRINTF("Moving to next row for window id %d\n", next_window->header.id);
        // Input loading offsets
        next_window->header.tile_info.input_horizontal_offset = 0;
        next_window->header.tile_info.input_vertical_offset += (window->header.tile_info.processed.loaded_input_rows - kernel_size + 1);
        next_window->header.tile_info.input_vertical_offset_index += (window->header.tile_info.processed.loaded_input_rows - kernel_size + 1) * config.config.input_feature_width;

        // Output storing offsets
        next_window->header.tile_info.output_horizontal_offset = 0;
        next_window->header.tile_info.output_vertical_offset += window->header.tile_info.output_height * config.config.output_feature_width;

        // Reset the window height/width
        
        // if (kernel_size > 0) {
        next_window->header.tile_info.output_height = (TILE_HEIGHT - kernel_size + 1);
        next_window->header.tile_info.output_width = (TILE_WIDTH - kernel_size + 1);
        // } else {
        //     next_window->header.tile_info.output_height = TILE_HEIGHT;
        //     next_window->header.tile_info.output_width = TILE_WIDTH;
        // }

        // Reset the input index offset for the next input feature
        next_window->run_config.config.input_index_offset = 0;
        // Reset the kernel index offset to the beginning of the kernels for this output feature
        next_window->run_config.config.kernel_index_offset -= (next_window->run_config.config.kernel_index_offset % next_window->run_config.config.inputs_per_output);

        // We've actually gone through the entire input feature
        // So we need to move to the next output feature and reset the tile offsets
        if (next_window->header.tile_info.input_vertical_offset >= config.config.input_feature_height - kernel_size + 1) {
            // Reset input offsets
            next_window->header.tile_info.input_horizontal_offset = 0;
            next_window->header.tile_info.input_vertical_offset = 0;
            next_window->header.tile_info.input_vertical_offset_index = 0;

            // Reset output offsets
            next_window->header.tile_info.output_horizontal_offset = 0;
            next_window->header.tile_info.output_vertical_offset = 0;

            next_window->header.output_feature_index += 1;
            next_window->run_config.config.kernel_index_offset += next_window->run_config.config.inputs_per_output;

            DPU_PRINTF("[Governor] Been here\n");

            // Do we need to go to the next output feature
            if (next_window->run_config.config.input_index_offset >= next_window->run_config.config.inputs_per_output - 1) {
                DPU_PRINTF("[Governor] Completed all tiles for output feature index %d\n", next_window->header.output_feature_index);

                if (next_window->run_config.config.kernel_index_offset >= (next_window->run_config.config.num_outputs * next_window->run_config.config.inputs_per_output)) {
                    DPU_PRINTF("[Governor] Completed all output features for run config\n");
                    return 2;
                }

                return 1;
            }

            // DPU_PRINTF("[Governor] I think the entire input has been processed... moving to the next output feature: %d\n", next_window->header.output_feature_index);
        }
    }

    return 0;
}


/**
 * Load the kernel for a given window
 */
void load_kernel_for_window(struct window_t *window) {

    uint16_t kernel_index = window->run_config.config.start_kernel_index_offset + window->run_config.config.kernel_index_offset;
    uint8_t kernel_size = config.config.kernel_size;
    // kernel_size = kernel_size * kernel_size; // Flatten the kernel size for easier indexing
    __mram_ptr int8_t *fused_ss_kernel_ptr = get_fused_ss_kernel_ptr(kernel_index);

    // I know this isn't that great of a copy but i became lazy
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int8_t fused_ss_value = *fused_ss_kernel_ptr;
            // Decode the signs and shifts
            window->kernel.shifts[ky][kx] = fused_ss_value & 0x3F;
            window->kernel.signs[ky][kx] = ((int8_t)(fused_ss_value & 0xC0)) >> 6;
            fused_ss_kernel_ptr++;
        }
    }
}

/**
 * Load the window with data and add padding where neccesary
 */
static void load_window(struct window_t *window, uint16_t window_id) {
    uint8_t padding = config.config.padding;
    uint8_t opcode = config.config.opcode;

    // We work from the assumption that we don't have padding
    uint16_t inputHeight = MIN(
        TILE_HEIGHT,
        config.config.input_feature_height - window->header.tile_info.input_vertical_offset
    );

    // Copy smaller row widths when we are at the right edge of the input feature map
    uint16_t inputWidth = MIN(
        TILE_WIDTH,
        config.config.input_feature_width - window->header.tile_info.input_horizontal_offset
    );

    uint16_t windowHeight = inputHeight;
    uint16_t windowWidth = inputWidth;

    // Find the padding where needed
    bool padTop = false;
    bool padBottom = false;
    bool padLeft = false;
    bool padRight = false;

    if (padding != 0) {

        // If we have padding on any side, that will reduce the number of rows/columns we need to copy
        // Horizontal padding
        if (window->header.tile_info.input_vertical_offset == 0) {
            padTop = true;
            // It could be that with padding, the window isn't large enough anymore
            if (windowHeight + padding > TILE_HEIGHT) {
                inputHeight -= padding;
            } else {
                windowHeight += padding;
            }
        }
        // If we have less rows to copy than the tile height, we are at the bottom edge
        // Orrrr; the input feature is too small to fill one entire tile..
        if (windowHeight < TILE_HEIGHT) {
            padBottom = true;
            // It could be that with padding, the window isn't large enough anymore
            if (windowHeight + padding > TILE_HEIGHT) {
                inputHeight -= padding;
            } else {
                windowHeight += padding;
            }
        }
        // Vertical padding
        if (window->header.tile_info.input_horizontal_offset == 0) {
            padLeft = true;
            // It could be that with padding, the window isn't large enough anymore
            if (windowWidth + padding > TILE_WIDTH) {
                inputWidth -= padding;
            } else {
                windowWidth += padding;
            }
        }
        // The offset is indexed from 0, hence + 1
        if (window->header.tile_info.input_horizontal_offset + windowWidth + 1 >= config.config.input_feature_width) {
            padRight = true;
            // It could be that with padding, the window isn't large enough anymore
            if (windowWidth + padding > TILE_WIDTH) {
                inputWidth -= padding;
            } else {
                windowWidth += padding;
            }
        }
    }

    DPU_PRINTF("Padding for: top: %d, bottom: %d, left: %d, right: %d\n", padTop, padBottom, padLeft, padRight);

    window->header.id = window_id;
    window->header.free = false;

    // Transform to uint8 to use the hardware accelerator
    window->header.payload_size = windowHeight * windowWidth;

    // The output size doesn't change with padding, just less rows are loaded from the payload
    window->header.tile_info.width = windowWidth;
    window->header.tile_info.height = windowHeight;
    if (shouldPerformOP(OP_CONV2D, opcode)) {
        uint8_t kernel_size = config.config.kernel_size;
        window->header.tile_info.output_width = windowWidth - kernel_size + 1;
        window->header.tile_info.output_height = windowHeight - kernel_size + 1;
    } else if (shouldPerformOP(OP_MAXPOOL, opcode)) {
        uint8_t kernel_size = config.config.kernel_size;
        uint8_t stride = config.config.stride;
        window->header.tile_info.output_width = ((windowWidth - kernel_size) / stride) + 1;
        window->header.tile_info.output_height = ((windowHeight - kernel_size) / stride) + 1;
    } else {
        window->header.tile_info.output_width = windowWidth;
        window->header.tile_info.output_height = windowHeight;
    }

    memset(window->output_alignment_padding, 0, sizeof(window->output_alignment_padding));
    memset(window->alignment_padding, 0, sizeof(window->alignment_padding));

    DPU_PRINTF("Loading window id %d: payload_size: %d, tile width: %d, tile height: %d, output width: %d, output height: %d, output horizontal offset: %d\n",
        window->header.id,
        window->header.payload_size,
        window->header.tile_info.width,
        window->header.tile_info.height,
        window->header.tile_info.output_width,
        window->header.tile_info.output_height,
        window->header.tile_info.output_horizontal_offset
    );
    DPU_PRINTF("Input feature[%d][%d], input horizontal offset: %d, input vertical offset: %d\n",
        window->header.output_feature_index,
        window->run_config.config.input_index_offset,
        window->header.tile_info.input_horizontal_offset,
        window->header.tile_info.input_vertical_offset
    );

    uint16_t offset = 0;
    // Copy all the individual rows with padding
    uint16_t row = 0;

    // Set padding for the top row
    if (padTop) {
        for (int i = 0; i < window->header.tile_info.width * padding; i += window->header.tile_info.width) {
            memset(&window->payload[i], 0, window->header.tile_info.width * sizeof(fxp32_t));
            offset += window->header.tile_info.width;
        }

        // Otherwise the alignment correction will be off
        row += padding;
    }

    // Set padding for the first row
    // if (padLeft) {
    //     memset(&window->payload[offset], 0, PADDING * sizeof(fxp32_t));
    //     // DPU_PRINTF("Set padding to the left at payload[%d]\n", offset);
    //     offset += PADDING;
    // }

    // The inputsize excludes horizontal padding
    uint32_t startIndex = window->header.tile_info.input_horizontal_offset + window->header.tile_info.input_vertical_offset_index;
    uint32_t endIndex = startIndex + (inputHeight * config.config.input_feature_width);

    uint32_t absolute_input_feature_index = window->run_config.config.start_input_index_offset
        + window->header.output_feature_index * window->run_config.config.input_stride
        + window->run_config.config.input_index_offset;
    for (int i = startIndex; i < endIndex; i += config.config.input_feature_width) {
        // Set the left padding
        if (padLeft) {
            memset(&window->payload[offset], 0, padding * sizeof(fxp32_t));
            // DPU_PRINTF("Set padding at payload[%d] for row %d\n", offset, row);
            offset += padding;
        }

        uint32_t desiredDestination = (uint32_t) &window->payload[offset];
        uint32_t sourceAddress = (uint32_t) &get_input_ptr(absolute_input_feature_index)[i];

        // uint32_t restorableValue = window->payload[offset - 1];
        uint32_t restorableValue = 0;

        // DMA transfers happen fine as long as both source and destination are aligned the same way
        // It can copy a bit too much though (before and after)
        // if (modulo8(desiredDestination) == modulo8(sourceAddress)) {
        if (modulo8(desiredDestination) == 0) {
            // No alignment compensation needed
            window->alignment_padding[row] = 0;
        } else {
            // Alignment compensation needed
            // Fix the alignment, and correct the value later
            restorableValue = window->payload[offset - 1];
            offset += 1;
            window->alignment_padding[row] = 1;
        }


        desiredDestination = (uint32_t) &window->payload[offset];

        // uint32_t destAlignment = modulo8(desiredDestination);
        // uint32_t sourceAlignment = modulo8(sourceAddress);
        // uint32_t first_value = get_input_ptr(absolute_input_feature_index)[i];

        // Copy the actual input row
        DMATransfer_MRAM_to_WRAM(&get_input_ptr(absolute_input_feature_index)[i], 
            &window->payload[offset],
            inputWidth * sizeof(fxp32_t)
        );

        // Restore the overwritten value
        if (window->alignment_padding[row] != 0) {
            window->payload[offset - 1] = restorableValue;
        }

        offset += inputWidth;

        // Set the right padding
        if (padRight) {
            memset(&window->payload[offset], 0, padding * sizeof(fxp32_t));
            // DPU_PRINTF("Set padding at payload[%d] for row %d\n", offset, row);
            offset += padding;
        }

        row++;
    }

    // Calculate the output alignments
    uint32_t outputIndex = 0;
    uint32_t output_offset = 0;
    for (int y = 0; y < window->header.tile_info.output_height; y++) {

        uint32_t mramDestination = (uint32_t) &get_output_ptr(window->header.output_feature_index)[
                window->header.tile_info.output_horizontal_offset
                + window->header.tile_info.output_vertical_offset
                + output_offset // next row
            ];

        uint32_t wramSource = (uint32_t)&window->output[
                outputIndex
            ];

        // DMA transfers happen fine as long as both source and destination are aligned the same way
        if (modulo8(wramSource) == modulo8(mramDestination)) {
            // No alignment compensation needed
            // DPU_PRINTF("No alignment compensation needed for window id %d, row %d\n", window->header.id, y);
            window->output_alignment_padding[y] = 0;
        } else {
            // Alignment compensation needed
            // DPU_PRINTF("Alignment compensation NEEDED for window id %d, row %d\n", window->header.id, y);
            window->output_alignment_padding[y] = 1;
        }
        
        outputIndex += window->header.tile_info.output_width;
        outputIndex += window->output_alignment_padding[y];
        // Removing this multiplication? Bam -40ms
        output_offset += config.config.output_feature_width;
        
    }

    // Set the output to 0, this is needed for the accumulation of the convolution results
    if (window->run_config.config.input_index_offset == 0) {
        memset(&window->output, 0, WINDOW_PAYLOAD_SIZE * sizeof(fxp32_t));
        DPU_PRINTF("[Governor] Input index offset is 0, setting output to 0 for window id %d\n", window->header.id);
    }

    // We've set a bit too much padding at the end, set back this offset 
    offset -= padding;

    // Set padding for the bottom rows
    if (padBottom) {
        memset(&window->payload[offset], 0, padding * windowWidth * sizeof(fxp32_t));
    }

    // Set some offsets for the next window
    window->header.tile_info.processed.loaded_input_cols = inputWidth;
    window->header.tile_info.processed.loaded_input_rows = inputHeight;

    // Load kernels for the window:
    load_kernel_for_window(window);

    // We've loaded this input id for this window
    window->run_config.config.input_index_offset += 1;
    // But we also need to move the kernel index offset, as the kernels are indexed on both input and output feature
    window->run_config.config.kernel_index_offset += 1;

    // Check if this is the last input for this run, if so we can set a flag such the workers can do relu
    if (!has_inputs_left_to_process(window)) {
        DPU_PRINTF("[Governor] No more inputs left to process for window id %d, output feature index %d\n", window->header.id, window->header.output_feature_index);
        // Set a flag in the window header, so the workers know they can do relu
        window->header.last_window_before_saving = true;
    } else {
        window->header.last_window_before_saving = false;
    }

    DPU_PRINTF("Loaded window id %d \n", window->header.id);
}

/**
 * Load the first window into WRAM
 */
static void load_first_window(struct governor_t *governor) {
    init_tile(&windows[0], 
        /* window_id= */1,
        /* run_config_index= */0);
    load_window(&windows[0], 1);

    // Ids start from 1, indexes from 0
    governor->latest_prepared_window.id = 1;
    governor->latest_prepared_window.window_index = 0;

    DPU_PRINTF("Governor loaded first window into WRAM\n");
}

/**
 * Initialize the governor and prepare the first window
 */
void governor_initialize(struct governor_t *governor) {
    // Copy the config:
    DMATransfer_MRAM_to_WRAM(&mconfig, &config, sizeof(struct config_padded_t));

    __mram_ptr struct run_config_t *run_config = get_run_configs_ptr(0);
    uint16_t windows_per_input = ((config.config.input_feature_width + 2 * config.config.padding + (TILE_WIDTH - 1)) / TILE_WIDTH) * ((config.config.input_feature_height + 2 * config.config.padding + (TILE_HEIGHT - 1)) / TILE_HEIGHT);
    uint16_t total_op_count = 0;

    for (uint16_t i = 0; i < config.config.num_configs; i++) {
        __mram_ptr struct run_config_t *run_config = get_run_configs_ptr(i);
        // DPU_PRINTF("Run config: \n");
        // DPU_PRINTF("  start_input_index_offset: %d\n", run_config->start_input_index_offset);
        // DPU_PRINTF("  input_index_offset: %d\n", run_config->input_index_offset);
        // DPU_PRINTF("  inputs_per_output: %d\n", run_config->inputs_per_output);
        // DPU_PRINTF("  start_kernel_index_offset: %d\n", run_config->start_kernel_index_offset);
        // DPU_PRINTF("  kernel_index_offset: %d\n", run_config->kernel_index_offset);
        // DPU_PRINTF("  num_outputs: %d\n", run_config->num_outputs);
        // DPU_PRINTF("  input_stride: %d\n", run_config->input_stride);
        total_op_count += (run_config->inputs_per_output * run_config->num_outputs);
    }

    // for (int8_t i = 0; i < MAX_OUTPUT_FEATURES; i++) {
    //     convolution_count += (config.run_configs[i].kernel_count * config.run_configs[i].input_count);
    // }

    *governor = (struct governor_t){
        .latest_prepared_window = {0, 0},
        .latest_processed_window = {0, 0},
        .latest_saved_window = {0, 0},
        .total_windows_allocated = {
            .id = total_op_count * windows_per_input,
            .window_index = 0
        },
        .workers_released = false,
    };

    DPU_PRINTF("[Governor] Initialized with total_windows_allocated: %d (total_op_count: %d, windows_per_input: %d)\n",
        governor->total_windows_allocated.id,
        total_op_count,
        windows_per_input
    );

    DPU_PRINTF("[Governor] Printing regular config: \n");
    DPU_PRINTF("Num configs: %d, num inputs: %d\n",
        config.config.num_configs,
        config.config.num_inputs
    );
    DPU_PRINTF("Num outputs: %d\n", config.config.num_outputs);
    DPU_PRINTF("Num kernels: %d\n", config.config.num_kernels);
    // DPU_PRINTF("Num runs: %d\n", config.config.num_runs);

    DPU_PRINTF("Offsets for run configs: \n");
    DPU_PRINTF("Run config: %04x\n", config.config.run_config_base_bytes);
    DPU_PRINTF("Input buffer: %04x\n", config.config.input_base_bytes);
    DPU_PRINTF("Output buffer: %04x\n", config.config.output_base_bytes);
    // DPU_PRINTF("Shift kernel buffer: %04x\n", config.config.shiftkernel_base_bytes);
    // DPU_PRINTF("Sign kernel buffer: %04x\n", config.config.signkernel_base_bytes);

    DPU_PRINTF("[Governor] Printing run config: \n");
    // for (int i = 0; i < config.config.num_configs; i++) {
    //     __mram_ptr struct run_config_t *run_config = get_run_configs_ptr(i);
    //     DPU_PRINTF("Run config %d: start_input_index_offset: %d, start_kernel_index_offset: %d, input_count: %d, kernel_count: %d\n",
    //         i,
    //         run_config->start_input_index_offset,
    //         run_config->start_kernel_index_offset,
    //         run_config->input_count,
    //         run_config->kernel_count
    //     );
    // }

    DPU_PRINTF("[Governor] Total of %d windows to process\n", governor->total_windows_allocated.id);

    // Mark all the windows as saved, so the governor knows they are free to use
    for (int i = 0; i < NUM_WINDOWS; i++) {
        windows[i].header.free = true;
    }

    // Load the first window so workers can start working immediately
    load_first_window(governor);

    // Release the workers
    DPU_PRINTF("[Governor] Releasing workers\n");
    barrier_wait(&barrier);
    profiling_stop(&governor_boot);
}

void findLastFullyProcessedWindow(struct window_position_t *window) {
    window->id = UINT32_MAX;
    for (int i = 0; i < NR_WORKERS; i++) {
        if (tasklets[i].last_processed_window.id < window->id) {
            window->id = tasklets[i].last_processed_window.id;
        }
    }

    // We can't use the value from the tasklet, as its not guaranteed to be up to date
    // As it can be a race condition, we need to find the window index in WRAM ourselves
    for (uint8_t i = 0; i < NUM_WINDOWS; i++) {
        if (windows[i].header.id == window->id) {
            window->window_index = i;
        }
    }

    return;
}

/**
 * Goveror working function
 */
void governor_work(struct governor_t *governor) {
    DPU_PRINTF("[Governor] Started working\n");
    bool fullyProcessed = false;

    while (!fullyProcessed) {
        struct window_position_t lastFullyProcessed;
        
        findLastFullyProcessedWindow(&lastFullyProcessed);

        DPU_PRINTF("[Governor] Last fully processed window id: %d\n", lastFullyProcessed.id);

        // The workers have completed a window for all inputs, transfer back to MRAM
        if (governor->latest_saved_window.id < lastFullyProcessed.id && lastFullyProcessed.id > 0) {
            // The window has been processed, but not for all inputs, we can prepare the next tile for this window without saving back to MRAM yet
            if (has_inputs_left_to_process(&windows[lastFullyProcessed.window_index])) {
                // So we still kind of want to say the window is free
                windows[lastFullyProcessed.window_index].header.free = true;
                incrementWindowPosition(&governor->latest_saved_window);
                DPU_PRINTF("[Governor] Window %d has been fully processed for one input, but has more inputs to process, marking WRAM window[%d] as free and prepared for the next input\n",
                    lastFullyProcessed.id,
                    lastFullyProcessed.window_index
                );
            } else {
                DPU_PRINTF("[Governor] Saving window %d back to MRAM from processed WRAM window[%d]\n",
                    lastFullyProcessed.id,
                    lastFullyProcessed.window_index
                );
                struct window_t *window = &windows[lastFullyProcessed.window_index];

                uint32_t outputIndex = 0;
                uint32_t output_offset = 0;
                if (window->header.id != 0) {
                    for (int y = 0; y < window->header.tile_info.output_height; y++) {

                        // Alignment compensation
                        outputIndex += window->output_alignment_padding[y];

                        uint32_t desiredDestination = (uint32_t) &get_output_ptr(window->header.output_feature_index)[
                                window->header.tile_info.output_horizontal_offset
                                + window->header.tile_info.output_vertical_offset
                                + output_offset // next row
                            ];

                        uint32_t sourceAddress = (uint32_t) &window->output[
                                outputIndex
                            ];
                        
                        // Source alignment
                        uint8_t sourceAlignment = modulo8(sourceAddress);
                        uint8_t destAlignment = modulo8(desiredDestination);
                        uint8_t misalignment = sourceAlignment ^ destAlignment;

                        DPU_PRINTF("[Governor] Saving row %d of window id %d from output[%d] to output_buffer[%d], sourceAlignment: %d, destAlignment: %d, misalignment: %d\n",
                            y,
                            window->header.id,
                            outputIndex,
                            window->header.output_feature_index,
                            sourceAlignment,
                            destAlignment,
                            misalignment
                        );

                        // Rule for dma transfers: source, destination and size modulo 8 must be 0 for all
                        DMATransfer_WRAM_to_MRAM(
                            (void *) sourceAddress,
                            (__mram_ptr void *) desiredDestination,
                            window->header.tile_info.output_width * sizeof(fxp32_t)
                        );

                        outputIndex += window->header.tile_info.output_width;
                        output_offset += config.config.output_feature_width;
                    }
                }
                // Free up the input and output windows
                windows[lastFullyProcessed.window_index].header.free = true;

                incrementWindowPosition(&governor->latest_saved_window);
                DPU_PRINTF("[Governor] Fully processed window %d saved back to MRAM\n", governor->latest_saved_window.id);

                // If that was the last window, we are done
                if (governor->latest_saved_window.id >= governor->total_windows_allocated.id) {
                    // DPU_PRINTF("Governor has saved all windows\n");
                    governor->workers_released = true;
                    fullyProcessed = true;
                }
            }
        }

        // There are still windows to process, prepare the next one if possible
        if (governor->latest_prepared_window.id < governor->total_windows_allocated.id) {
            uint8_t next_input_index = nextWindowIndex(governor->latest_prepared_window.window_index);
            struct window_t *next_window = &windows[next_input_index];
            struct window_t *last_prepared_window = &windows[governor->latest_prepared_window.window_index];

            // uint8_t next_output_index = nextWindowIndex(windows[governor->latest_prepared_window.window_index].header.output_window_index);
            if (!next_window->header.free) {
                // Cannot prepare the next window as the next windows in WRAM is still occupied
                DPU_PRINTF("[Governor] Cannot prepare next window as WRAM window[%d] (id: %d) is still occupied\n", next_input_index, governor->latest_prepared_window.id + 1);
                continue;
            }

            // We have a free window, we need to prepare the next window always
            prep_next_window(last_prepared_window, next_window);

            // It doesn't have inputs left to process, so we can prepare the next tile and load the new payload for the next tile
            if (!has_inputs_left_to_process(next_window)) {
                bool moved_to_new_input = prep_next_tile(last_prepared_window, next_window);
                
                
                load_window(next_window, next_window->header.id);
                DPU_PRINTF("[Governor] Prepared next window id %d for new tile in WRAM window[%d] (next tile return code: %d)\n",
                    windows[next_input_index].header.id,
                    next_input_index,
                    moved_to_new_input
                );
            // The window is partially processed, so we just need to load the new input
            } else {
                load_window(next_window, next_window->header.id);

                DPU_PRINTF("[Governor] Prepared next window id %d in WRAM window[%d] with new input for partially processed tile\n",
                    windows[next_input_index].header.id,
                    next_input_index
                );
            }

            incrementWindowPosition(&governor->latest_prepared_window);
        } else {
            // DPU_PRINTF("[Governor] All windows have been prepared\n");
        }
    }
    DPU_PRINTF("[Governor] Finished processing all windows, released workers\n");
}
