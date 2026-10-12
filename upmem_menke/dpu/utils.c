#include "utils.h"
#include <string.h>
#include "mram_unaligned.h"


/**
 * Returns the next window index in a circular manner
 */
uint8_t nextWindowIndex(uint8_t current_index) {
    if (current_index >= NUM_WINDOWS - 1) {
        return 0;
    } else {
        return current_index + 1;
    }
}

/**
 * Returns the previous window index in a circular manner
 */
uint8_t previousWindowIndex(uint8_t current_index) {
    if (current_index == 0) {
        return NUM_WINDOWS - 1;
    } else {
        return current_index - 1;
    }
}

/**
 * Increments both the absolute and wraparound window position counters
 */
void incrementWindowPosition(struct window_position_t *position) {
    // Increment absolute counter
    position->id += 1;

    // Increment wraparound counter
    position->window_index = nextWindowIndex(position->window_index);
}

/**
 * DMA transfers are limited to 2048 bytes at a time.
 * This is just a function rename for clarity.
 */
void DMATransfer_MRAM_to_WRAM (
    __mram_ptr void const *mram_src,
    void *wram_dst,
    uint16_t size_bytes
) {
    mram_read_unaligned(
        mram_src,
        wram_dst,
        size_bytes);
}

/**
 * DMA transfers are limited to 2048 bytes at a time.
 * This is just a function rename for clarity.
 */
void DMATransfer_WRAM_to_MRAM (
    void const *wram_src,
    __mram_ptr void *mram_dst,
    uint16_t size_bytes
) {
    mram_write_unaligned(
        wram_src,
        mram_dst,
        size_bytes
    );
}

// __mram_ptr int8_t *get_shiftkernel_ptr(uint16_t kernel_idx) {
//     return (__mram_ptr int8_t *)&mram_arena[
//         config.config.shiftkernel_base_bytes + kernel_idx * (KERNEL_SIZE * KERNEL_SIZE * sizeof(int8_t))
//     ];
// }

// __mram_ptr int8_t *get_signkernel_ptr(uint16_t kernel_idx) {
//     return (__mram_ptr int8_t *)&mram_arena[
//         config.config.signkernel_base_bytes + kernel_idx * (KERNEL_SIZE * KERNEL_SIZE * sizeof(int8_t))
//     ];
// }