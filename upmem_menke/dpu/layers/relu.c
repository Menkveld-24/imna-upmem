
#include "relu.h"

void doReluForWindowAfterConv2D(
    const struct tasklet_t *tasklet,
    struct window_t *working_window
) {
    int tid = tasklet->id;                 // worker ID
    int stride = NR_WORKERS;               // number of workers

    uint32_t size = working_window->header.tile_info.output_width * working_window->header.tile_info.output_height;
    // Add an extra row in case of alignmentcompensation
    size += working_window->header.tile_info.height;
    // But we also don't want to go out of bounds
    size = MAX(size, TILE_WIDTH * TILE_HEIGHT);

    for (int i = tid; i < size; i += stride) {
        if (working_window->output[i] < 0) {
            working_window->output[i] = 0;
        }
    }
}