import subprocess
import sys


if len(sys.argv) < 3:
    print("Usage: python run_10_times.py <times> <script.py> [args...]")
    sys.exit(1)

TIMES=int(sys.argv[1])

for i in range(TIMES):
    print(f"--- Run {i + 1}/{TIMES} ---")
    subprocess.run([sys.executable] + sys.argv[2:])
