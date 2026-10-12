DPU_CC := dpu-upmem-dpurte-clang

NR_TASKLETS      ?= 11
PRINTF_ENABLED   ?= 0
DPU_OPTIMIZATION ?= -O3

DPU_DEFINES ?= -DNR_TASKLETS=$(NR_TASKLETS) -g -fstack-size-section $(DPU_OPTIMIZATION) \
               -DPRINTF_ENABLED=$(PRINTF_ENABLED)

BUILD_DIR := build

DPU_BIN_A := $(BUILD_DIR)/bin/dpu_a
DPU_BIN_B := $(BUILD_DIR)/bin/dpu_b
DPU_BIN_C := $(BUILD_DIR)/bin/dpu_c
DPU_BIN_D := $(BUILD_DIR)/bin/dpu_d
DPU_BIN_E := $(BUILD_DIR)/bin/dpu_e
DPU_BIN_F := $(BUILD_DIR)/bin/dpu_f
DPU_BIN_G := $(BUILD_DIR)/bin/dpu_g

DPU_ASM_A := $(BUILD_DIR)/asm/dpu_a.s
DPU_ASM_B := $(BUILD_DIR)/asm/dpu_b.s
DPU_ASM_C := $(BUILD_DIR)/asm/dpu_c.s
DPU_ASM_D := $(BUILD_DIR)/asm/dpu_d.s
DPU_ASM_E := $(BUILD_DIR)/asm/dpu_e.s
DPU_ASM_F := $(BUILD_DIR)/asm/dpu_f.s
DPU_ASM_G := $(BUILD_DIR)/asm/dpu_g.s

.PHONY: all clean

all: $(DPU_BIN_A) $(DPU_BIN_B) $(DPU_BIN_C) $(DPU_BIN_D) $(DPU_BIN_E) $(DPU_BIN_F) $(DPU_BIN_G) \
     $(DPU_ASM_A) $(DPU_ASM_B) $(DPU_ASM_C) $(DPU_ASM_D) $(DPU_ASM_E) $(DPU_ASM_F) $(DPU_ASM_G)

$(DPU_BIN_A): dpu/implementationA.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_B): dpu/implementationB.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_C): dpu/implementationC.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_D): dpu/implementationD.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_E): dpu/implementationE.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_F): dpu/implementationF.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_BIN_G): dpu/implementationG.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $< -o $@

$(DPU_ASM_A): dpu/implementationA.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_B): dpu/implementationB.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_C): dpu/implementationC.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_D): dpu/implementationD.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_E): dpu/implementationE.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_F): dpu/implementationF.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@

$(DPU_ASM_G): dpu/implementationG.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -S $< -o $@


clean:
	rm -rf $(BUILD_DIR)
