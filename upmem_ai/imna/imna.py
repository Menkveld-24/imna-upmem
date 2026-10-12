import torch
from imna.dputensor import DpuTensor


class IMNABridge:
    def __init__(self, dpu_binary: str, dpu_limit: int = 64):
        self._bridge = torch.classes.upmem.IMNABridge(dpu_limit, dpu_binary)
        self._layers: list = []
        self._built = False

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------

    def _register_layer(self, layer) -> None:
        self._layers.append(layer)

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def build(self, input_shape: tuple) -> None:
        """Compute the per-DPU MRAM layout for all registered layers and
        stamp each layer with its offsets.

        input_shape : (N, C, H, W) of the first layer's input.
        """
        N, C, H, W = input_shape

        inputWidth=W
        inputHeight=H
        for layer in self._layers:
            # Calculate the layer's input and output sizes based on the input dimensions, and update the descriptor
            inputWidth, inputHeight = layer.calculateLayerSize(input_width=inputWidth, input_height=inputHeight)
             # Align the sizes to 8 bytes
            layer.descriptor.alignSizes()


        # run_config is always at mram_arena[0];
        # Ignore it as we'll deal with it in te C code
        MRAM_SIZE = 62 * 1024 * 1024   # 62 MiB usable MRAM per DPU
        MRAM_SIZE -= 1024   # reserve 1 KiB for run_config and other metadata, just to be safe

        input_location = 0
        weights_location = 0
        output_location = MRAM_SIZE

        # in/outputs are hugging the edges of the MRAM arena, weights are in the middle
        for index, layer in enumerate(self._layers):
            # Even layers get their inputs from the left edge, odd layers from the right edge
            if (index%2 == 0):
                input_location = 0
                output_location = MRAM_SIZE - layer.descriptor.getOutputBytes()
                weights_location = input_location + layer.descriptor.getInputBytes()
            else:
                # The current layer's input is the output location of the previous layer
                input_location = output_location
                output_location = 0
                weights_location = output_location + layer.descriptor.getOutputBytes()

            layer.descriptor.input_offset_bytes = input_location
            layer.descriptor.output_offset_bytes = output_location
            layer.descriptor.weight_offset_bytes = weights_location

        for index, layer in enumerate(self._layers):
            print(f"Layer {index}: input_offset={layer.descriptor.input_offset_bytes}, "
                  f"output_offset={layer.descriptor.output_offset_bytes}, "
                  f"weight_offset={layer.descriptor.weight_offset_bytes}, "
                  f"input_bytes={layer.descriptor.getInputBytes()}, "
                  f"output_bytes={layer.descriptor.getOutputBytes()}, "
                  f"weight_bytes={layer.descriptor.getWeightBytes()}")
        
            size = layer.descriptor.getInputBytes() + layer.descriptor.getOutputBytes() + layer.descriptor.getWeightBytes()
            if size > MRAM_SIZE:
                raise ValueError(f"Layer {index} exceeds MRAM size with total bytes {size} (input {layer.descriptor.getInputBytes()}, output {layer.descriptor.getOutputBytes()}, weights {layer.descriptor.getWeightBytes()})")

        self._built = True

    def download(self, dpu_tensor: DpuTensor) -> torch.Tensor:
        """Download a DpuTensor from DPU MRAM to CPU."""
        return self._bridge.download(dpu_tensor._tensor)