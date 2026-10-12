#ifndef WINDOWED_H
#define WINDOWED_H

#include "mram.h"
#include "stdint.h"
#include "stdbool.h"
#include <profiling.h>

#ifndef PRINTF_ENABLED
    #define PRINTF_ENABLED 1
#endif

#if PRINTF_ENABLED
    #include <stdio.h>
    #define DPU_PRINTF(...) printf(__VA_ARGS__)
#else
    #define DPU_PRINTF(...)
#endif

#ifndef RESTORE_MRAM
    #define RESTORE_MRAM 1
#endif

// Define the input feature map
// #define padding 1
// #ifndef PADDING
//     #define PADDING 1
// #endif

// #ifndef STRIDE
//     #define STRIDE 1
// #endif

// This defines a 3x3 kernel
#define MAX_KERNEL_SIZE 7
// #ifndef KERNEL_SIZE
//     #define KERNEL_SIZE 3
// #endif
// This means the kernel has x fields on each side of the center
// #define kernel_fields_from_center ((KERNEL_SIZE - 1) / 2)

// #ifndef FEATURE_SIZE
//     #define FEATURE_SIZE 28
// #endif

// #define input_features 1
// #define input_feature_width FEATURE_SIZE
// #define input_feature_height FEATURE_SIZE

// #define PADDED_INPUT_FEATURE_SIZE (input_feature_width + (2 * PADDING)) * (input_feature_height + (2 * PADDING))

// Output feature map is dependent on the kernel and padding (larger kernel = smaller output)
// #define output_feature_width  ((input_feature_width - KERNEL_SIZE + (PADDING * 2)) + 1)
// #define output_feature_height ((input_feature_height - KERNEL_SIZE + (PADDING * 2)) + 1)

// #define MAX_WRAM_BYTES 65536 // 64KB - V1A
#define MAX_WRAM_BYTES 63488 // 64KB - V1B
#define MAX_MRAM_BYTES 64 * 1024 * 1024 // 64MB
// 5MB reserved for kernels and config

// Each output is the result of the summed convolution of one filter with y inputs
// So the number of output features is equal to the number of filters, and the number of input features is the number of feature maps we convolve over
// #define MAX_OUTPUT_FEATURES 10
// #define MAX_INPUT_FEATURES 640
// #define MAX_KERNEL_COUNT

#if RESTORE_MRAM
#define ARENA_SIZE_BYTES 20000
#else
#define ARENA_SIZE_BYTES MAX_MRAM_BYTES - (2 * 1024 * 1024)
#endif
// ====================== Memory layout explanation =====================
//      MRAM layout:
// 
// In case output > input, the end of both input and output are aligned
// So output[-1][-1] aligns with input[-1][-1]
// 
// Output feature map 
// |-------------------|
// |___________________|
// |Input feature map  |
// |                   |
// |                   |  
// |-------------------|
// 
// If the output feature map is smaller than the input feature map, the input feature map is used directly
// so output[0][0] aligns with input[0][0]
// 
//     WRAM layout:
// The WRAM keeps a circular buffer divided into windows that are the size of the kernel height * input width
// This way, convolutions can be performed in a streaming row-by-row fashion.
// There are multiple windows, The current window, 

// WRAM is 64kb, window size should be kernel_size * max(input_feature_width, output_feature_width) 
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
// Make sure it is aligned to 8 bytes for DMA
#define MULTIPLEOF8BYTES(x) (((x) + 7) & ~7)
// #define INPUT_FEATURE_SIZE (input_feature_width * input_feature_height)
// #define OUTPUT_FEATURE_SIZE (output_feature_width * output_feature_height)

#define NUM_WINDOWS 2
// #define MAX_WINDOWS_BYTES 40000
// #define WINDOW_PAYLOAD_SIZE ((MAX_WINDOWS_BYTES / NUM_WINDOWS) / sizeof(fxp32_t))
#define WINDOW_PAYLOAD_SIZE (2500)

// The tile width is exclusive of alignment compensation
#define TILE_WIDTH 48
#define TILE_HEIGHT (WINDOW_PAYLOAD_SIZE / (TILE_WIDTH + 1))

// #define TILE_OUTPUT_HEIGHT (TILE_HEIGHT - KERNEL_SIZE + 1)
// #define TILE_OUTPUT_WIDTH (TILE_WIDTH - KERNEL_SIZE + 1)

// Number of worker tasklets (excluding the governor)
#define NR_WORKERS (NR_TASKLETS - 1)

struct window_position_t {
    // This is a wraparound counter for the window index
    uint32_t window_index;
    // This is an absolute counter for the window index
    uint32_t id;
};

typedef int32_t fxp32_t;

struct tasklet_t {
    uint8_t id;
    struct window_position_t last_processed_window;
    struct window_position_t working_window;
};

struct tile_position_t {
    uint16_t _1D_index;
    // The position of the current tile in the input feature map
    uint16_t row_index;
    uint16_t col_index;
    // The number of cols and rows this input feature is broken into
    uint16_t cols;
    uint16_t rows;

    // The total number of tiles for this input feature map
    uint16_t num_tiles;
};

struct processed_tile_stats_t {
    uint16_t loaded_input_rows;
    uint16_t loaded_input_cols;
};

struct tile_t {
    uint16_t id;

    // How large is the operating width and height of a tile
    uint16_t width;
    uint16_t height;

    // These are the effective dimensions after accounting for the kernel size
    uint16_t output_width;
    uint16_t output_height;

    // This is the offset for loading inputs
    uint32_t input_vertical_offset_index;

    // This is the offset in rows/columns
    uint32_t input_horizontal_offset;
    uint32_t input_vertical_offset;
    
    // This is the offset for saving outputs back to MRAM
    uint16_t output_horizontal_offset;
    uint16_t output_vertical_offset;

    // This keeps track of how much actually has been loaded in this tile
    struct processed_tile_stats_t processed;
};

struct kernel_info_t {
    __dma_aligned int8_t shifts[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];      // the kernel weights
    __dma_aligned int8_t signs[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];      // the kernel weights
};

struct window_header_t {
    uint32_t id;                  // the window’s unique identifier
    bool last_window_before_saving;
    uint16_t output_feature_index;
    bool free;             // whether this window has been saved back to MRAM
    // payload
    uint16_t payload_start; // This is the start of the payload
    uint16_t payload_size;      // payload size
    struct tile_t tile_info;   // tile information
};

struct governor_t {
    // This keeps track of how much work there is still to do
    struct window_position_t total_windows_allocated;
    // This keeps track of the last window that has been prepared for workers
    struct window_position_t latest_prepared_window;
    // This keeps track of the latest processed window by workers
    struct window_position_t latest_processed_window;
    // This keeps track of the last window that has been saved from workers
    struct window_position_t latest_saved_window;

    // Keep the workers active
    bool workers_released;
};

struct run_config_t {
    uint16_t start_input_index_offset;
    uint16_t input_index_offset;
    uint16_t inputs_per_output;   // how many inputs accumulate into each output (C_in for conv, 1 for maxpool)
    uint16_t start_kernel_index_offset;
    uint16_t kernel_index_offset;
    uint16_t num_outputs;         // number of output channels covered by this run config
    uint16_t input_stride;        // input base shift per output (0 = shared inputs like conv, 1 = per-channel like maxpool)
};

struct run_config_padded_t {
    struct run_config_t config;
    uint8_t padding[
        MULTIPLEOF8BYTES(sizeof(struct run_config_t)) - sizeof(struct run_config_t)
    ];
};

struct config_t {
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

struct config_padded_t {
    struct config_t config;
    uint8_t padding[
        MULTIPLEOF8BYTES(sizeof(struct config_t)) - sizeof(struct config_t)
    ];
};

// struct progress_tracker_t {
//     uint16_t total_inputs_to_process;
//     uint16_t next_input_index;
// };

struct window_t {
    struct window_header_t header; // the window’s header
    struct kernel_info_t kernel;
    // struct progress_tracker_t progress_tracker;
    __dma_aligned struct run_config_padded_t run_config;
    uint8_t alignment_padding[55];
    uint8_t output_alignment_padding[55];
    fxp32_t __dma_aligned payload[WINDOW_PAYLOAD_SIZE];   // the window’s payload
    fxp32_t __dma_aligned output[WINDOW_PAYLOAD_SIZE];    // the window’s output payload
};

#endif