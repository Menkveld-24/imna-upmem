#pragma once
#include <cstdint>
#include "dpu_dtypes.h"

void write_timing_to_csv(
    const char      *csv_file,
    int64_t          batch_id,
    const Context *ctx,
    double tensor_alloc_ms,
    double build_dist_ms,
    double dpu_alloc_ms,
    double dpu_load_ms,
    double xfer_config_ms,
    double sg_xfer_ms,
    double dpu_launch_ms,
    double xfer_results_ms,
    double total_ms
);
