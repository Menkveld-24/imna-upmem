import time
import torch
import torch.nn as nn
from imna.utils import multipleof8
from imna import IMNABridge, DpuTensor
from imna.layerdescriptor import LayerDescriptor, MustContainLayerDescriptor

class Conv2D(nn.Module, MustContainLayerDescriptor):
    def __init__(self,
                 bridge: IMNABridge,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 applyReluAfter=True,
                 dpu_logs=True):
        super().__init__()

        self.bridge       = bridge
        self.in_channels  = input_channels
        self.out_channels = output_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.applyReluAfter = applyReluAfter
        self.dpu_logs     = dpu_logs
        self.t_quantize   = 0.0
        self.t_dpu        = 0.0

        self.register_buffer('fused_ss',
            torch.zeros(output_channels, input_channels, kernel_size, kernel_size, dtype=torch.int8))

        bridge._register_layer(self)

    def calculateLayerSize(self, input_width: int, input_height: int) -> tuple[int, int]:
        Hout = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wout = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # *4 because we're using int32_t as a datatype
        input_bytes = 4 * self.in_channels * input_width * input_height
        output_bytes = 4 * self.out_channels * Wout * Hout

        self.descriptor = LayerDescriptor(
            # int8_t weights, for shift and sign
            weight_bytes=2 * multipleof8(1 * self.out_channels * self.in_channels * self.kernel_size * self.kernel_size),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
        self._Hout = Hout
        self._Wout = Wout
        return Wout, Hout


    def forward(self, x: DpuTensor) -> DpuTensor:
        if not isinstance(x, DpuTensor):
            raise TypeError("Input must be a DpuTensor, got {}".format(type(x)))
        if not self._built():
            raise RuntimeError("Call handle.build(input_shape) before the first forward pass")


        t = time.perf_counter()
        result_handle = self.bridge._bridge.mram_conv2d(
            x.unwrap(),
            self.fused_ss.contiguous(),
            self.stride,
            self.padding,
            self.applyReluAfter,
            self.descriptor.input_offset_bytes,
            self.descriptor.output_offset_bytes,
            self.descriptor.weight_offset_bytes
        )
        self.t_dpu = time.perf_counter() - t
        return DpuTensor._from_handle(result_handle)

    def _built(self) -> bool:
        return hasattr(self, 'descriptor') and hasattr(self, '_Hout')


class MaxPooling(nn.Module, MustContainLayerDescriptor):
    def __init__(self,
                 bridge: IMNABridge,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__()

        self.bridge         = bridge
        self.channels       = channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding

        bridge._register_layer(self)

    def calculateLayerSize(self, input_width: int, input_height: int) -> tuple[int, int]:
        Hout = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        Wout = (input_width  + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.descriptor = LayerDescriptor(
            weight_bytes=0,
            input_bytes=4 * self.channels * input_width * input_height,
            output_bytes=4 * self.channels * Wout * Hout,
        )
        self._Hout = Hout
        self._Wout = Wout
        return Wout, Hout

    def forward(self, x: DpuTensor) -> DpuTensor:
        if not isinstance(x, DpuTensor):
            raise TypeError("Input must be a DpuTensor, got {}".format(type(x)))
        if not self._built():
            raise RuntimeError("Call handle.build(input_shape) before the first forward pass")

        result_handle = self.bridge._bridge.mram_maxpooling(
            x.unwrap(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.descriptor.input_offset_bytes,
            self.descriptor.output_offset_bytes,
            self.descriptor.weight_offset_bytes,
        )
        return DpuTensor._from_handle(result_handle)

    def _built(self) -> bool:
        return hasattr(self, 'descriptor') and hasattr(self, '_Hout')
