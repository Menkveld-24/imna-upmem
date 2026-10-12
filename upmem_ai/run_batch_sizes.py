import subprocess
import sys

BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2000]
NUM_BATCHES = 15

for batch_size in BATCH_SIZES:
    print(f"\n=== Batch size: {batch_size} ===")
    subprocess.run([
        sys.executable, "measure.py",
        "--batch-size", str(batch_size),
        "--num-batches", str(NUM_BATCHES),
    ])
