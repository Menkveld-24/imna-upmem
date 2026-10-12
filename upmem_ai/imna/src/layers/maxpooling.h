#ifndef _MAXPOOLING_H_
#define _MAXPOOLING_H_

#include <stdint.h>
#include <vector>
#include "../dpu_dtypes.h"
#include "../dputensor.h"
#include "../distribution.h"
#include "../dpu_pool.h"

DpuTensor mram_maxpool_impl(
    DpuTensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    const struct MramPartitioning *mram_partitioning,
    DpuPool& dpu_pool
);

#endif
