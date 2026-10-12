import csv
import os
import random
import numpy as np
from input_feature_gen import InputFeatureGenerator
from dpu import DpuSet

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'bin')
N_RUNS = 80
INPUT_WIDTH = 80
INPUT_HEIGHT = 80

IMPLEMENTATIONS = {
    'Deepshift Optimized': {'binary': f'{BUILD_DIR}/dpu_a', 'type': 'shifts'},
    'Int naive': {'binary': f'{BUILD_DIR}/dpu_b', 'type': 'int_kernel'},
    'Float naive': {'binary': f'{BUILD_DIR}/dpu_c', 'type': 'int_kernel'},
    'Int addernet': {'binary': f'{BUILD_DIR}/dpu_d', 'type': 'int_kernel'},
    'Float addernet': {'binary': f'{BUILD_DIR}/dpu_e', 'type': 'int_kernel'},
    'No alignment compensation DP': {'binary': f'{BUILD_DIR}/dpu_f', 'type': 'shifts'},
    'DeepShift Baseline': {'binary': f'{BUILD_DIR}/dpu_g', 'type': 'baseline'},
}

SIGN_VARIANTS = ['dense', 'realistic']

random.seed(42)

rows = []

for sign_type in SIGN_VARIANTS:
    random.seed(42)
    print(f"\n=== Sign variant: {sign_type} ===")

    for run in range(1, N_RUNS + 1):
        seed = random.randint(1, 0xFFFFFFFF)
        print(f"Run {run}/{N_RUNS}  seed=0x{seed:08X}")

        for impl, config in IMPLEMENTATIONS.items():
            if config['type'] == 'shifts' or config['type'] == 'baseline':
                feature = InputFeatureGenerator(
                    kernel_size=3, feature_width=INPUT_WIDTH, feature_height=INPUT_HEIGHT,
                    padding=1, seed=seed,
                )
            else:
                feature = InputFeatureGenerator(
                    kernel_size=3, feature_width=INPUT_WIDTH, feature_height=INPUT_HEIGHT,
                    padding=1, kernel_datatype=np.int32, seed=seed,
                )

            with DpuSet(nr_dpus=1, binary=config['binary']) as dpu:
                dpu.payload = feature.prepare_dpu_input()
                if config['type'] == 'shifts':
                    dpu.input_shifts = feature.prepare_dpu_shifts_kernel()
                    if sign_type == 'dense':
                        dpu.input_signs = feature.prepare_dpu_dense_signs_kernel()
                    else:
                        dpu.input_signs = feature.prepare_dpu_realistic_signs_kernel()
                elif config['type'] == 'baseline':
                    dpu.input_shifts = feature.prepare_dpu_baseline_shifts_kernel()
                    if sign_type == 'dense':
                        dpu.input_signs = feature.prepare_dpu_dense_signs_kernel()
                    else:
                        dpu.input_signs = feature.prepare_dpu_realistic_signs_kernel()
                else:
                    dpu.input_kernel = feature.prepare_dpu_int_kernel()

                dpu.exec()
                cycles = dpu.nb_cycles.uint32()

            rows.append({
                'run': run, 'seed': hex(seed), 'sign_type': sign_type,
                'implementation': impl, 'cycles': cycles,
            })
            print(f"  {impl}: {cycles} cycles")

with open('measurements.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['run', 'seed', 'sign_type', 'implementation', 'cycles'])
    writer.writeheader()
    writer.writerows(rows)

print("Saved to measurements.csv")
