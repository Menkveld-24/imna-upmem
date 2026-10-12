from imna.utils import multipleof8

# Layer descriptor keeps track of the size per layer, is per-batch!
class LayerDescriptor:
    weight_bytes: int
    input_bytes: int
    output_bytes: int

    output_offset_bytes: int
    input_offset_bytes: int
    weight_offset_bytes: int

    def __init__(self, weight_bytes, input_bytes, output_bytes):
        self.weight_bytes = weight_bytes
        self.input_bytes = input_bytes
        self.output_bytes = output_bytes
        
        self.output_offset_bytes = 0
        self.input_offset_bytes = 0
        self.weight_offset_bytes = 0

    def alignSizes(self):
        self.input_bytes = multipleof8(self.input_bytes)
        self.output_bytes = multipleof8(self.output_bytes)
        self.weight_bytes = multipleof8(self.weight_bytes)
    
    def getWeightBytes(self) -> int:
        return self.weight_bytes
    
    def getInputBytes(self) -> int:
        return self.input_bytes

    def getOutputBytes(self) -> int:
        return self.output_bytes

class MustContainLayerDescriptor:
    descriptor :LayerDescriptor

    def calculateLayerSize(self, input_width: int, input_height: int) -> tuple[int, int]:
        raise NotImplementedError("Must implement calculateLayerSize to compute the layer's input and output bytes based on the input dimensions.")

    def getLayerDescriptor(self) -> LayerDescriptor:
        return self.descriptor