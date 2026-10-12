#pragma once
#include <cstdio>
#include <cstring>
#include <torch/library.h>
#include "dpu_includes.h"
#include "dpu_dtypes.h"
#include <torch/library.h>

struct DpuPool {
    dpu_set_t set = {};
    uint32_t nr_dpus = 0;
    bool loaded = false;

    void allocate(uint32_t requested_dpus) {
        if (nr_dpus == requested_dpus) return;

        if (nr_dpus > 0) {
            dpu_free(set);
            nr_dpus = 0;
            loaded = false;
        }

        if (requested_dpus < 5) {
            printf("Requesting too few DPUs (%u) for sg transfers, setting \"sgXferMaxBlocksPerDpu=5\"...\n", requested_dpus);
            DPU_ASSERT(dpu_alloc(requested_dpus, "sgXferEnable=true,sgXferMaxBlocksPerDpu=5", &set));
        } else {
            DPU_ASSERT(dpu_alloc(requested_dpus, "sgXferEnable=true", &set));
        }
        DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
    }

    void load(const char *dpu_binary) {
        if (loaded) return;

        DPU_ASSERT(dpu_load(set, dpu_binary, NULL));
        loaded = true;
    }

    void readyCheck() {
        TORCH_CHECK(loaded, "DPU pool is not ready: no binary loaded");
    }

    ~DpuPool() {
        if (nr_dpus > 0) {
            printf("Freeing DPU set with %u DPUs\n", nr_dpus);
            dpu_free(set);
        }
    }
};
