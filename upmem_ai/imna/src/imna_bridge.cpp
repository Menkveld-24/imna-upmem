#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/intrusive_ptr.h>
#include <string_view>
#include <torch/library.h>
#include "dpu_includes.h"
#include "dpu_pool.h"
#include "dputensor.h"
#include "distribution.h"
#include "layers/conv2d.h"
#include "layers/maxpooling.h"

struct IMNABridge : torch::CustomClassHolder {
    DpuPool dpu_pool;

    explicit IMNABridge(int64_t dpu_limit, std::string_view dpu_binary) {
        printf("Starting IMNABridge with DPU limit: %ld\n", dpu_limit);
        dpu_pool.allocate((uint32_t) dpu_limit);
        dpu_pool.load(dpu_binary.data());
    }

    ~IMNABridge() {
        printf("Destroying IMNABridge\n");
    }

    c10::intrusive_ptr<DpuTensor> mram_conv2d(
        c10::intrusive_ptr<DpuTensor> input,
        at::Tensor fused_ss,
        int64_t stride,
        int64_t padding,
        bool applyReluAfter,
        int64_t input_offset_bytes,
        int64_t output_offset_bytes,
        int64_t weight_offset_bytes
    ) {

        MramPartitioning partitioning;
        createPartitioning(&partitioning, input_offset_bytes, output_offset_bytes, weight_offset_bytes);
        
        DpuTensor out = mram_conv2d_impl(
            *input,
            fused_ss,
            stride,
            padding,
            applyReluAfter,
            &partitioning,
            dpu_pool
        );
        return c10::make_intrusive<DpuTensor>(std::move(out));
    }

    c10::intrusive_ptr<DpuTensor> mram_maxpooling(
        c10::intrusive_ptr<DpuTensor> input,
        int64_t kernel_size,
        int64_t stride,
        int64_t padding,
        int64_t input_offset_bytes,
        int64_t output_offset_bytes,
        int64_t weight_offset_bytes
    ) {

        MramPartitioning partitioning;
        createPartitioning(&partitioning, input_offset_bytes, output_offset_bytes, weight_offset_bytes);
        
        DpuTensor out = mram_maxpool_impl(
            *input,
            kernel_size,
            stride,
            padding,
            &partitioning,
            dpu_pool
        );
        return c10::make_intrusive<DpuTensor>(std::move(out));
    }

    at::Tensor eager_conv2d(
        at::Tensor input,
        at::Tensor fused_ss,
        int64_t stride,
        int64_t padding,
        bool applyReluAfter
    ) {
        DpuTensor out = eager_conv2d_impl(
            input,
            fused_ss,
            stride,
            padding,
            applyReluAfter,
            dpu_pool
        );
        return this->download(c10::make_intrusive<DpuTensor>(std::move(out)));
    }

    at::Tensor download(c10::intrusive_ptr<DpuTensor> x) {
        at::Tensor output = at::empty(
            x->data_sizes,
            at::TensorOptions().dtype(at::kInt)
        );

        int32_t* out_ptr = output.data_ptr<int32_t>();
        dpu_set_t dpu;
        uint32_t dpu_index = 0;

        // Each DPU contains just 1 batch
        uint32_t output_size = 1 * x->data_sizes[1] * x->data_sizes[2] * x->data_sizes[3];

        DPU_FOREACH(dpu_pool.set, dpu, dpu_index) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, out_ptr + dpu_index * output_size));
        }

        DPU_ASSERT(dpu_push_xfer(
            dpu_pool.set,
            DPU_XFER_FROM_DPU,
            "mram_arena",
            x->mram_location,
            output_size * sizeof(int32_t),
            DPU_XFER_DEFAULT));


        return output;
    }
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
TORCH_LIBRARY(upmem, m) {
    m.class_<DpuTensor>("DpuTensor")
     .def(torch::init<at::Tensor>())
     .def("sizes",   &DpuTensor::sizes)
     .def("tensor",  &DpuTensor::tensor)
     .def("onCPU",   &DpuTensor::onCPU)
     .def("onDPU",   &DpuTensor::onDPU)
     .def_readwrite("data_sizes",    &DpuTensor::data_sizes)
     .def_readwrite("mram_location", &DpuTensor::mram_location);

    m.class_<IMNABridge>("IMNABridge")
     .def(torch::init<int64_t, std::string_view>())
     .def("download",       &IMNABridge::download)
     .def("eager_conv2d",         &IMNABridge::eager_conv2d)
     .def("mram_conv2d",    &IMNABridge::mram_conv2d)
     .def("mram_maxpooling",    &IMNABridge::mram_maxpooling);
}
