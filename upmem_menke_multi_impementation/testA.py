import os
from input_feature_gen import InputFeatureGenerator
from dpu import DpuSet
import subprocess
import time
from sys import stdout

input_feature_width=80
input_feature_height=80

feature = InputFeatureGenerator(kernel_size=3, feature_width=input_feature_width, feature_height=input_feature_height, padding=1, seed=0xC0FFEE)

with DpuSet(nr_dpus=1, binary=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'bin', 'dpu_a')) as dpu:


    # dpu.input_buffer = feature.prepare_dpu_input()
    # dpu.mconfig = todpumconfig
    # dpu.mkernels = feature.prepare_dpu_kernel()
    # Print the inputs for debugging
    print("Shifts:")
    print(feature.shifts)
    print("Signs:")
    print(feature.signs)

    # Load the inputs
    dpu.payload = feature.prepare_dpu_input()
    dpu.input_shifts = feature.prepare_dpu_shifts_kernel()
    dpu.input_signs = feature.prepare_dpu_signs_kernel()

    t0 = time.perf_counter()
    dpu.exec(log=stdout)
    # dpu.exec()
    t1 = time.perf_counter()
    print(f"dpu.exec wall time: {(t1 - t0) * 1000:.3f} ms")
    cycles = dpu.nb_cycles.uint32()
    print(f"Cycles taken for convolution: {cycles}")
    clocks_per_sec = dpu.CLOCKS_PER_SEC.uint32()
    print(f"Clocks per second: {clocks_per_sec}")
    time_sec = cycles / clocks_per_sec
    print(f"Time taken for convolution: {time_sec:.6f} seconds")

    # value = dpu.output_buffer.uint32()[0:feature.feature_height*feature.feature_width*num_items]
    # conv_cpu = feature.runConv()

    # feature.compare(dpu_output=value, conv_output=conv_cpu)