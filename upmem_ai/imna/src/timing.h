#pragma once
#include <time.h>

static inline double ms_since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1e3
         + (now.tv_nsec - start.tv_nsec) * 1e-6;
}

#define TICK(var) struct timespec var; clock_gettime(CLOCK_MONOTONIC, &var)
#define TOCK(var, label) PRINTF(stderr, "  [accel] %-30s %7.2f ms\n", label, ms_since(var))
