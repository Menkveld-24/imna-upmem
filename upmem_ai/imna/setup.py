# from distutils.core import setup
from setuptools import setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys

print("Building imna_bridge extension...")
# Versions:
print("PyTorch version:", torch.__version__)
print("Python version:", sys.version)

setup(
    name='imna_bridge',
    ext_modules=[
        CppExtension(
            name='dpu._C',
            sources=[
                './src/imna_bridge.cpp',
                './src/layers/conv2d.cpp',
                './src/layers/maxpooling.cpp',
                './src/distribution.cpp',
                './src/sg_xfer.cpp',
                './src/csv_timing.cpp',
                # './src/old_accel.cpp'
            ],
            extra_compile_args=[
                '-I/usr/include/dpu',
            ],
            extra_link_args=[
                "-ldpu",
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
