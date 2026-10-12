#ifndef _CONV2D_H_
#define _CONV2D_H_

#include <stdint.h>
#include <vector>
#include "../dpu_dtypes.h"
#include "../dputensor.h"
#include "../distribution.h"
#include "../dpu_pool.h"


DpuTensor eager_conv2d_impl(
    at::Tensor input,
    at::Tensor fused_ss,
    int64_t stride,
    int64_t padding,
    bool applyReluAfter,
    DpuPool& dpu_pool
);

DpuTensor mram_conv2d_impl(
    DpuTensor input,
    at::Tensor fused_ss,
    int64_t stride,
    int64_t padding,
    bool applyReluAfter,
    const struct MramPartitioning *mram_partitioning,
    DpuPool& dpu_pool
);

#endif
