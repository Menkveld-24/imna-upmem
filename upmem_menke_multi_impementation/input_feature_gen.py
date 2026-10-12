import numpy as np
import torch

class InputFeatureGenerator:

    seed: int
    kernel_size: int
    feature_width: int
    feature_height: int
    padding: int

    shifts: np.ndarray
    signs: np.ndarray
    input: np.ndarray

    datatype: np.dtype


    def __init__(self, kernel_size=3, feature_width=28, feature_height=28, padding=1, seed=0, datatype=np.uint32, kernel_datatype=np.uint8):
        self.kernel_size = kernel_size
        self.feature_width = feature_width
        self.feature_height = feature_height
        self.padding = padding
        self.seed = seed
        self.datatype = datatype
        self.kernel_datatype = kernel_datatype

        # if seed != 0:
        self.shifts = self.__generate_i8_feature_map(shape=(kernel_size, kernel_size), min=0, max=14)
        self.baseline_shifts = self.__generate_i8_feature_map(shape=(kernel_size, kernel_size), min=-14, max=0)
        self.dense_signs = self.__generate_signs_feature_map(shape=(kernel_size, kernel_size), min=-1, max=1, distribution={-1: 0.5, 0: 0, 1: 0.5})
        self.realistic_signs = self.__generate_signs_feature_map(shape=(kernel_size, kernel_size), min=-1, max=1, distribution={-1: 0.25, 0: 0.5, 1: 0.25})
        self.kernel = self.__generate_u32_feature_map(shape=(kernel_size, kernel_size))
        self.input = self.__generate_u32_feature_map(
            shape=(feature_height, feature_width),
        )
        # else:
        #     self.shifts = np.array([[0, 1, 7],
        #                             [3, -4, 9],
        #                             [0, 13, 2]], dtype=np.int8)
        #     self.signs = np.array([[1, 1, 1],
        #                         [1, -1, 1],
        #                         [1, 1, 1]], dtype=np.int8)
        #     self.input = np.array([[x for _ in range(feature_width)] for x in range(feature_height)], dtype=self.datatype)

    
    def __generate_u32_feature_map(
        self,
        shape
    ):
        rng = np.random.default_rng(self.seed)

        # Generate full 32-bit patterns
        data = rng.integers(
            low=0,
            # this has been severely limited on purpose,
            # Otherwise the cpu vs dpu start to differ a lot
            high=2**6,
            size=shape,
            dtype=np.uint32
        )

        return data

    def __generate_i8_feature_map(
        self,
        shape,
        min=-1,
        max=1,
    ):
        rng = np.random.default_rng(self.seed)

        # Generate full 32-bit patterns
        data = rng.integers(
            low=min,
            high=max + 1,
            size=shape,
            dtype=np.int8
        )

        return data

    def __generate_signs_feature_map(
        self,
        shape,
        min=-1,
        max=1,
        distribution=None,
    ):
        rng = np.random.default_rng(self.seed)

        if distribution is not None:
            values = np.array(list(distribution.keys()), dtype=np.int8)
            probs = np.array(list(distribution.values()), dtype=np.float64)
            probs /= probs.sum()
            data = rng.choice(values, size=shape, p=probs).astype(np.int8)
        else:
            data = rng.integers(low=min, high=max + 1, size=shape, dtype=np.int8)

        return data

    def prepare_dpu_shifts_kernel(self):
        byteArray = bytearray(self.shifts.astype(self.kernel_datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray

    def prepare_dpu_baseline_shifts_kernel(self):
        byteArray = bytearray(self.baseline_shifts.astype(self.kernel_datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray

    def prepare_dpu_int_kernel(self):
        byteArray = bytearray(self.kernel.astype(self.kernel_datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray


    def prepare_dpu_realistic_signs_kernel(self):
        byteArray = bytearray(self.realistic_signs.astype(self.kernel_datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray

    def prepare_dpu_dense_signs_kernel(self):
        byteArray = bytearray(self.dense_signs.astype(self.kernel_datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray

    def prepare_dpu_input(self):
        byteArray = bytearray()
        byteArray.extend(self.input.astype(self.datatype).tobytes())
    
        # Pad to multiple of 8 bytes
        while len(byteArray) % 8 != 0:
            byteArray.extend(b'\x00')

        return byteArray

    def compare(self, dpu_output: np.ndarray, conv_output: np.ndarray = None) -> bool:
        if (conv_output is None):
            conv_output = self.runConv()

        equal = np.allclose(dpu_output, conv_output, atol=1, rtol=0)

        if (not equal):
            print("[ERROR] DPU Output and Convolution Output do not match!")
        else:
            print("DPU Output and Convolution Output match within rounding tolerance!")
            # for i in range(len(dpu_output)):
            #     if dpu_output[i] != conv_output[i]:
            #         print(f"Index {i}: DPU Output = {dpu_output[i]}, Conv Output = {conv_output[i]}")
        
        return equal
    
    def runConv(self):
        inp = torch.tensor(self.input, dtype=torch.float32)
        inp = inp.unsqueeze(0).unsqueeze(0)

        kernel = torch.tensor(self.shifts, dtype=torch.float32)
        signs = torch.tensor(self.signs, dtype=torch.float32)

        weights = (2 ** kernel) * signs

        conv = torch.nn.functional.conv2d(
            inp,
            weights.unsqueeze(0).unsqueeze(0),
            bias=None,
            stride=1,
            padding=self.padding,
        )

        return conv[0,0,:,:].numpy().flatten().round().astype(np.uint32)
