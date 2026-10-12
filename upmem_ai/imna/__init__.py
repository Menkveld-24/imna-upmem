import os
import glob
import torch

_here = os.path.dirname(__file__)

# If you built as upmem.dpu._C, the file will be upmem/dpu/_C*.so
_candidates = sorted(glob.glob(os.path.join(_here, "dpu", "_C*.so")))

if not _candidates:
    raise FileNotFoundError(
        "Could not find built extension in upmem/dpu/_C*.so\n"
        "Build it from repo root:\n"
        "  python -m pip install -v -e .\n"
        "or:\n"
        "  python setup.py build_ext --inplace -v\n"
    )

# Load the op-registration shared library *without importing it as a Python module*
torch.ops.load_library(_candidates[-1])

# Now safe to import Python wrappers that reference torch.ops.upmem.*
from .imna import IMNABridge, DpuTensor
from . import eager
from .mram import Conv2D, MaxPooling
from .modeltransformer import ModelTransformer