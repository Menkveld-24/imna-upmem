# Compiler
# HOST_CC := clang
DPU_CC  := dpu-upmem-dpurte-clang
HOST_CC  := gcc

HOST_INCLUDES := -I/workspaces/ubuntu-upmem/upmem_menke/upmem/build/host -I/usr/include/dpu

NR_TASKLETS ?= 11
PRINTF_ENABLED ?= 0
KERNEL_SIZE ?= 3
RESTORE_MRAM ?= 0
PADDING ?= 1
STRIDE ?= 1
FEATURE_SIZE ?= 28
DPU_OPTIMIZATION ?= -O3

DPU_COUNT ?= 128
HOST_NUM_FEATURES ?= 1

DPU_DEFINES ?= -DNR_TASKLETS=$(NR_TASKLETS) -g -fstack-size-section -pg $(DPU_OPTIMIZATION) -DPRINTF_ENABLED=$(PRINTF_ENABLED) -DRESTORE_MRAM=$(RESTORE_MRAM)
HOST_DEFINES ?= -DNUM_FEATURES=$(HOST_NUM_FEATURES) -DDPU_COUNT=$(DPU_COUNT) -DKERNEL_SIZE=$(KERNEL_SIZE) -DFEATURE_SIZE=$(FEATURE_SIZE) -DPADDING=$(PADDING) -DSTRIDE=$(STRIDE)

# Build dir
BUILD_DIR := build

# Output executables
HOST_BIN := $(BUILD_DIR)/bin/host
DPU_BIN  := $(BUILD_DIR)/bin/dpu
DPU_BIN_HDR = $(BUILD_DIR)/host/dpu_binary.h

# Source files
HOST_SRC := $(wildcard host/*.c)
DPU_SRC  := $(wildcard dpu/*.c) $(wildcard dpu/layers/*.c)

# Object files in mirrored build structure
HOST_OBJ := $(patsubst host/%.c,$(BUILD_DIR)/host/%.o,$(HOST_SRC))
DPU_OBJ  := $(patsubst dpu/%.c,$(BUILD_DIR)/dpu/%.o,$(DPU_SRC))

.PHONY: all clean run

all: $(HOST_BIN) $(DPU_BIN)

########################################
#   Build Host Program
########################################

# HOST_CFLAGS += -O2

# Create the dpu binary header 
# $(DPU_BIN_HDR): $(DPU_BIN)
# 	mkdir -p $(dir $@)
# 	xxd -i $(DPU_BIN) | sed 's/build_bin_dpu/dpu_binary/g' > $@

# Generate the host object files
$(BUILD_DIR)/host/%.o: host/%.c #$(DPU_BIN_HDR)
	mkdir -p $(dir $@)
	$(HOST_CC) -I$(BUILD_DIR)/host -O1 $(HOST_INCLUDES) $(HOST_DEFINES) -c $< -o $@

# Create the host binary from host object files and DPU binary
$(HOST_BIN): $(HOST_OBJ) $(DPU_BIN)
	mkdir -p $(dir $@)
	$(HOST_CC) -O1 $(HOST_DEFINES) \
	    $(HOST_OBJ) \
	    `dpu-pkg-config --libs --cflags dpu` \
	    -o $@

# DPU_BIN_SRC := $(BUILD_DIR)/host/dpu_binary.c
# DPU_BIN_OBJ := $(BUILD_DIR)/host/dpu_binary.o

# $(DPU_BIN_SRC): $(DPU_BIN)
# 	mkdir -p $(dir $@)
# 	echo "#include <stdint.h>" > $@
# 	xxd -i $(DPU_BIN) | sed 's/build_bin_dpu/dpu_binary/g' >> $@

# $(DPU_BIN_OBJ): $(DPU_BIN_SRC)
# 	$(HOST_CC) -c $< -o $@

# $(HOST_BIN): $(HOST_OBJ) $(DPU_BIN_OBJ) $(DPU_BIN)

# $(HOST_BIN): $(HOST_OBJ) $(DPU_BIN_HDR) $(DPU_BIN)
# 	mkdir -p $(dir $@)
# 	$(HOST_CC) -O0 \
# 	    $(HOST_OBJ) \
# 	    `dpu-pkg-config --libs --cflags dpu` \
# 	    -o $@

# $(DPU_BIN_HDR): $(DPU_BIN)
# 	xxd -i $(DPU_BIN) | sed 's/build_bin_dpu/dpu_binary/g' > $@

# `dpu-pkg-config --cflags --libs dpu`
########################################
#   Build DPU Program
########################################

$(DPU_BIN): $(DPU_OBJ)
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) $(DPU_SRC) -o $@


########################################
#   Build Rules for Object Files
########################################

# Compile host -> build/host/*.o
# $(BUILD_DIR)/host/%.o: host/%.c
# 	mkdir -p $(dir $@)
# 	$(HOST_CC) -c $< -o $@

# Compile DPU -> build/dpu/*.o
$(BUILD_DIR)/dpu/%.o: dpu/%.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -c $< -o $@

# Compile DPU layers -> build/dpu/layers/*.o
$(BUILD_DIR)/dpu/layers/%.o: dpu/layers/%.c
	mkdir -p $(dir $@)
	$(DPU_CC) $(DPU_DEFINES) -Idpu -c $< -o $@

########################################
#   Utility rules
########################################

clean:
	rm -rf $(BUILD_DIR)

run: $(HOST_BIN)
	./$(HOST_BIN)
