#pragma once
#include <cstdint>
#include "dpu_includes.h"

// bool configure_block(sg_block_info *out, uint32_t dpu_index, uint32_t block_index, void *args);
void dump_mram(struct dpu_set_t set, uint8_t *buffer, uint8_t *config_buffer, uint32_t buffer_size, uint32_t config_buffer_size);
void dump_mram_to_header(struct dpu_set_t dpu_set, uint64_t bytes);
