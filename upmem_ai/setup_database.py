"""
CSV-based experiment logger for upmem_ai.

Three files, all in output_dir
-------------------------------
runs.csv
    One row per training session.  run_id is a 1-based integer.

batches.csv
    One row per mini-batch (Python-side timings).
    batch_id is a 1-based integer.  run_id links back to runs.csv.

conv.csv
    One row per convolution_kernel() call, written directly from C++.
    batch_id links back to batches.csv.

Usage
-----
    from setup_database import CsvLogger

    log = CsvLogger("results")
    run_id = log.create_run(notes="mnist lr=0.01", batch_size=1024)

    # pass the shared conv.csv path to conv layers once at model construction:
    conv_csv = log.conv_csv_path()
    model.conv1.timing_csv_file = conv_csv
    model.conv2.timing_csv_file = conv_csv

    # inside training loop:
    batch_id = log.begin_batch(run_id, epoch=e, batch_idx=i)
    model.conv1.batch_id = batch_id
    model.conv2.batch_id = batch_id

    # ... forward / loss / backward / step ...

    log.finish_batch(batch_id, zero_grad_ms=..., forward_ms=..., ...)
"""

import csv
import os
from datetime import datetime
from typing import Optional

_RUNS_HEADER = ["run_id", "created_at", "batch_size", "notes"]

_BATCHES_HEADER = [
    "batch_id", "run_id", "epoch", "batch_idx",
    "zero_grad_ms", "forward_ms",
    "conv1_ms", "conv1_quantize_ms", "conv1_dpu_ms",
    "conv2_ms", "conv2_quantize_ms", "conv2_dpu_ms",
    "maxpool_ms", "fc_ms",
    "loss_ms", "backward_ms", "step_ms",
]

_CONV_HEADER = [
    "batch_id", "cin", "cout", "kernel_size", "batch_size",
    "tensor_alloc_ms", "build_dist_ms", "dpu_alloc_ms", "dpu_load_ms",
    "xfer_config_ms", "sg_xfer_ms", "dpu_launch_ms", "xfer_results_ms",
    "total_ms",
]


def _count_rows(path: str) -> int:
    """Return number of data rows (excluding header) in a CSV, or 0 if missing."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def _append_row(path: str, header: list, row: list) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


class CsvLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # initialise counters from existing files so IDs stay unique across restarts
        self._run_counter   = _count_rows(self._path("runs.csv"))
        self._batch_counter = _count_rows(self._path("batches.csv"))
        self._pending: dict[int, dict] = {}

    def _path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def conv_csv_path(self) -> str:
        """Path to the shared conv timings CSV — pass this to timing_csv_file."""
        return self._path("conv.csv")

    # --- runs ----------------------------------------------------------

    def create_run(
        self,
        notes: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        self._run_counter += 1
        run_id = self._run_counter
        _append_row(
            self._path("runs.csv"),
            _RUNS_HEADER,
            [run_id, datetime.now().isoformat(), batch_size, notes],
        )
        return run_id

    # --- batches -------------------------------------------------------

    def begin_batch(
        self,
        run_id: int,
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
    ) -> int:
        """Reserve the next batch_id and store metadata for finish_batch()."""
        batch_id = self._batch_counter
        self._batch_counter += 1
        self._pending[batch_id] = {"run_id": run_id, "epoch": epoch, "batch_idx": batch_idx}
        return batch_id

    def finish_batch(
        self,
        batch_id: int,
        *,
        zero_grad_ms: float = 0.0,
        forward_ms: float = 0.0,
        conv1_ms: float = 0.0,
        conv1_quantize_ms: float = 0.0,
        conv1_dpu_ms: float = 0.0,
        conv2_ms: float = 0.0,
        conv2_quantize_ms: float = 0.0,
        conv2_dpu_ms: float = 0.0,
        maxpool_ms: float = 0.0,
        fc_ms: float = 0.0,
        loss_ms: float = 0.0,
        backward_ms: float = 0.0,
        step_ms: float = 0.0,
    ) -> None:
        p = self._pending.pop(batch_id)
        _append_row(
            self._path("batches.csv"),
            _BATCHES_HEADER,
            [
                batch_id, p["run_id"], p["epoch"], p["batch_idx"],
                f"{zero_grad_ms:.3f}", f"{forward_ms:.3f}",
                f"{conv1_ms:.3f}", f"{conv1_quantize_ms:.3f}", f"{conv1_dpu_ms:.3f}",
                f"{conv2_ms:.3f}", f"{conv2_quantize_ms:.3f}", f"{conv2_dpu_ms:.3f}",
                f"{maxpool_ms:.3f}", f"{fc_ms:.3f}",
                f"{loss_ms:.3f}", f"{backward_ms:.3f}", f"{step_ms:.3f}",
            ],
        )

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def wipe(output_dir: str) -> None:
    """Delete the 3 CSV files in output_dir."""
    for name in ("runs.csv", "batches.csv", "conv.csv"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
    print(f"Wiped runs.csv, batches.csv, conv.csv in {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args    = sys.argv[1:]
    command = args[0] if args else "status"
    output_dir = args[1] if len(args) > 1 else "results"

    if command == "wipe":
        wipe(output_dir)
    elif command == "status":
        runs_path = os.path.join(output_dir, "runs.csv")
        if not os.path.exists(runs_path):
            print(f"No runs.csv found in {output_dir}")
            sys.exit(0)
        with open(runs_path, newline="") as f:
            runs = list(csv.DictReader(f))
        n_batches = _count_rows(os.path.join(output_dir, "batches.csv"))
        n_conv    = _count_rows(os.path.join(output_dir, "conv.csv"))
        print(f"{len(runs)} run(s), {n_batches} batch row(s), {n_conv} conv row(s)")
        for r in runs:
            print(f"  run {r['run_id']}  {r['created_at']}  batch_size={r['batch_size']}  notes={r['notes']!r}")
    else:
        print("Usage: python setup_database.py [status|wipe] [output_dir]")
        sys.exit(1)
