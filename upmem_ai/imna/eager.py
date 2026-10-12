import time
import torch
import torch.nn as nn
import imna.ste as ste
from imna.utils import fuseSignsAndShifts
from imna.imna import IMNABridge


class Conv2D(nn.Module):
    def __init__(self,
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    applyReluAfter=False,
                    bias=False,
                    rounding='deterministic',
                    act_integer_bits=16,
                    act_fraction_bits=16,
                    weight_bits=5
                    ):
        super().__init__()

        self.in_channels = input_channels
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.applyReluAfter = applyReluAfter
        self.output_padding = rounding
        self.rounding=rounding
        self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits

        self.shift_fp = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size), requires_grad=True)
        self.sign_fp = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.shift_fp.data.uniform_(*self.shift_range)
        self.sign_fp.data.uniform_(-1, 1)

    def forward(self, x: torch.Tensor, bridge: IMNABridge) -> torch.Tensor:
        # --- quantization (CPU) ---
        self.shift_fp.data = ste.clamp(self.shift_fp.data, *self.shift_range)
        shift_rounded = ste.round(self.shift_fp, self.rounding).to(torch.int8)
        sign_rounded = ste.sign(ste.round(self.sign_fp, self.rounding)).to(torch.int8)
        x = ste.round_fixed_point(x, self.act_integer_bits, self.act_fraction_bits).to(torch.int32)

        fused = fuseSignsAndShifts(sign_rounded.contiguous(), shift_rounded.contiguous())

        # --- DPU kernel ---
        out = bridge._bridge.eager_conv2d(
            x,
            fused,
            self.stride,
            self.padding,
            self.applyReluAfter
        )
        return out
