import torch

class DpuTensor:
    """Python wrapper around torch.classes.upmem.DpuTensor."""

    def __init__(self, x: torch.Tensor):
        self._tensor = torch.classes.upmem.DpuTensor(x)

    # No idea what this function does
    @classmethod
    def _from_handle(cls, handle):
        """Internal: wrap existing C++ DpuTensor"""
        obj = cls.__new__(cls)
        obj._tensor = handle
        return obj

    def sizes(self):
        return self._tensor.sizes()

    def tensor(self) -> torch.Tensor:
        return self._tensor.tensor()

    def onCPU(self) -> bool:
        return self._tensor.onCPU()
    
    def onDPU(self) -> bool:
        return self._tensor.onDPU()
    
    def getDPUlocation(self) -> int:
        return self._tensor.mram_location

    def unwrap(self):
        return self._tensor
    
    # def dtype(self):
    #     return self._tensor.dtype()

    def __repr__(self):
        return f"DpuTensor(sizes={self.sizes()})"