#ifndef RELU_H
#define RELU_H

#include "../dpu.h"


void doReluForWindowAfterConv2D(
    const struct tasklet_t *tasklet,
    struct window_t *working_window
);

#endif
