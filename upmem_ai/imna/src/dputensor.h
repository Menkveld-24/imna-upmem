#ifndef _DPUTENSOR_H_
#define _DPUTENSOR_H_

#include <ATen/ATen.h>
#include <torch/library.h>

// ---------------------------------------------------------------------------
// DpuTensor: a handle to tensor data living on the DPU side.
//
// Currently wraps a CPU at::Tensor. Future: replace `data` with an MRAM
// descriptor (DPU index + offset + shape) so activations never leave MRAM
// between layers.
// ---------------------------------------------------------------------------
struct DpuTensor : torch::CustomClassHolder {
    at::Tensor data;
    std::vector<long> data_sizes;

    // Whether the data has been uploaded to the DPU yet
    bool uploaded_to_dpu = false;
    // MRAM offset where the tensor data is located (valid after upload)
    int64_t mram_location = 0;

    DpuTensor() = default;

    // caffe2::TypeMeta _dtype;

    // Transfer the tensor ownership to DpuTensor
    explicit DpuTensor(at::Tensor t) {
        data = std::move(t);
        // _dtype = data.dtype();
        // Copy the sizes such that doesn't get lost when the original tensor is moved from
        data_sizes = data.sizes().vec();
    }

    bool onCPU() const {
        return !uploaded_to_dpu;
    }

    bool onDPU() const {
        return uploaded_to_dpu;
    }

    void uploadedToDpu(int64_t mram_location) {
        uploaded_to_dpu = true;
        this->mram_location = mram_location;
        data.reset();
    }

    std::vector<long> sizes() const {
        return data_sizes;
    }

    at::Tensor tensor() const {
        return data;
    }

    template<typename T>
    T* data_ptr() const {
        // return data.template.data_ptr<T>();
        if (this->onCPU()) {
            return data.data_ptr<T>();
        } else {
            // Data is on DPU, not accessible from CPU
            return nullptr;
        }
    }
};

#endif
