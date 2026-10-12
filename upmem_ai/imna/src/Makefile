CXX      := g++
CXXFLAGS := -std=c++17 -g -O0
DPU_CFLAGS  := $(shell dpu-pkg-config --cflags dpu)
DPU_LDFLAGS := $(shell dpu-pkg-config --libs dpu)

DPU_BINARY ?= $(abspath ../../upmem_menke/build/bin/dpu)

BUILD_DIR := build/host
TARGET    := $(BUILD_DIR)/sg_xfer_demo

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): sg_xfer_demo.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) \
	    -DDPU_BINARY='"$(DPU_BINARY)"' \
	    $(DPU_CFLAGS) \
	    $< -o $@ \
	    $(DPU_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
