# upmem_ai

`upmem_ai` runs the convolution and max-pooling layers of a trained
DeepShift-style neural network on UPMEM DPUs, from inside PyTorch. The rest of
the network (fully connected layers, softmax) stays on the CPU.

The core is **imna**, a small PyTorch extension:

- A C++ host bridge (`imna/src/`) that is compiled into `imna/dpu/_C*.so`. It
  registers the TorchScript classes `torch.classes.upmem.IMNABridge` and
  `torch.classes.upmem.DpuTensor`.
- Python wrappers (`imna/*.py`) that expose these as ordinary `nn.Module` layers.

The DPU side is the program in [`../upmem_menke`](../upmem_menke). imna loads
that compiled binary at runtime.

## How it works

1. **Weights.** A DeepShift layer stores each weight as a sign (-1, 0, +1) and a
   power-of-two shift. `ModelTransformer` fuses each sign/shift pair into a
   single `int8` value (`fused_ss`), which is the format the DPU kernel reads.
2. **Activations.** Inputs are quantized to 16.16 fixed point and stored as
   `int32` (`ste.round_fixed_point`).
3. **MRAM layout.** Layers register themselves with an `IMNABridge`.
   `bridge.build(input_shape)` computes where each layer's input, weights and
   output live in the 62 MiB of DPU MRAM. Consecutive layers alternate between
   the two ends of MRAM, so one layer's output stays on the DPU and becomes the
   next layer's input without a round trip to the host.
4. **Execution.** `imna.Conv2D` and `imna.MaxPooling` take and return a
   `DpuTensor` that stays resident on the DPUs. `bridge.download(t)` copies the
   final result back into a regular `torch.Tensor`. The batch is distributed
   over at most `dpu_limit` DPUs (the scripts set this to the batch size).

```python
import imna

bridge  = imna.IMNABridge(dpu_binary="../upmem_menke/build/bin/dpu", dpu_limit=batch_size)
conv1   = imna.Conv2D(bridge, 1, 32, kernel_size=3, padding=1, applyReluAfter=True)
conv2   = imna.Conv2D(bridge, 32, 64, kernel_size=3, padding=1, applyReluAfter=True)
maxpool = imna.MaxPooling(bridge, channels=64, kernel_size=2, stride=2)
bridge.build(input_shape=(batch_size, 1, 28, 28))

x = ste.round_fixed_point(images, 16, 16).to(torch.int32)
y = maxpool(conv2(conv1(imna.DpuTensor(x))))
features = bridge.download(y)   # back on the CPU
```

The weights (`conv1.fused_ss`, ...) are filled in by `load_state_dict` from a
fused checkpoint. See `mnist2.py` for the complete model.

`imna.eager.Conv2D` is an older, single-layer path. It quantizes on the CPU and
calls `eager_conv2d` for one layer at a time, without keeping data on the DPU.

## Setup

Run these inside the dev container (see the [root README](../README.md)), where
the UPMEM SDK is installed and `/usr/upmem_env.sh` has been sourced.

```bash
# 1. Build the DPU program that imna loads
make -C ../upmem_menke

# 2. Install Python dependencies in a virtual environment
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt

# 3. Build the imna C++ extension in place (produces imna/dpu/_C*.so)
./install.sh
```

Run the scripts from inside `upmem_ai/`. MNIST is downloaded to `../data` on
first use.

## Scripts

| Script | Purpose |
| --- | --- |
| `mnist2.py` | MNIST inference with conv1, conv2 and max-pooling on the DPUs. Prints accuracy and the average time per stage. |
| `measure.py` | Same model, used for the measurements. Accepts `--batch-size` and `--num-batches`, logs per-batch timings through `setup_database.CsvLogger`, and appends a row to `results/summary.csv`. |
| `run_batch_sizes.py` | Runs `measure.py` for batch sizes 32 to 2000. |
| `run_benchmark_x_times.py` | `python run_benchmark_x_times.py <times> <script.py> [args...]` repeats a script N times. |
| `transform_model.py` | `python transform_model.py <model.pt> [out.pt]` prints the tensors in a checkpoint and writes a copy with fused sign/shift weights (default name `<model>_fused.pt`). |
| `sign_distribution.py` | Prints how the sign weights (-1/0/+1) of a model are distributed, overall and per layer. |
| `setup_database.py` | `CsvLogger`, which writes `runs.csv`, `batches.csv` and `conv.csv` to a results folder. Run it directly to inspect or wipe a results folder. |
| `mnist.py`, `test.py` | Earlier experiments (training through the DPU, and a hand-written kernel test). They use layer classes that are no longer in `imna` and won't run as-is. |

## Models

| File | Contents |
| --- | --- |
| `mnist_shift_rounded.pt` | Trained MNIST shift network with separate sign and shift weights. |
| `mnist_shift_rounded_fused.pt` | The same model after `transform_model.py`. `mnist2.py` and `measure.py` load this one. |
| `mnist_shift_rounded_int8.pt` | Quantized `int8` sign/shift weights, used by `sign_distribution.py`. |

## Results

- `results/` contains the batch-size sweep produced by `measure.py`:
  - `runs.csv` has one row per run.
  - `batches.csv` has per-batch timings in milliseconds.
  - `summary.csv` has accuracy and the average time per stage in seconds.
- `measurements/28x28x3/` contains earlier timing logs from the C++ bridge
  (`timings_accel_*.csv`: allocation, load, transfer and launch time per layer)
  and from Python (`timings_python_*.csv`).
  - `baseline/` and `dpu_pooling/` compare running max-pooling on the CPU with
    running it on the DPU.
  - Each `N<batch size>` folder holds one batch size.

## Layout

```
imna/
  __init__.py          loads dpu/_C*.so and exports IMNABridge, DpuTensor, Conv2D, MaxPooling, ModelTransformer
  imna.py              IMNABridge: layer registration and MRAM layout
  mram.py              Conv2D and MaxPooling layers that keep data on the DPU
  eager.py             older single-layer Conv2D
  dputensor.py         DpuTensor wrapper
  layerdescriptor.py   per-layer byte sizes and MRAM offsets
  modeltransformer.py  sign/shift fusion for checkpoints
  ste.py, utils.py     quantization helpers, adapted from DeepShift
  setup.py             builds the C++ extension
  src/                 C++ bridge: DPU allocation, work distribution, scatter-gather transfers, timing
    layers/            host side of conv2d and maxpooling
```

## Credits

`imna/ste.py` and `imna/utils.py` are adapted from
[DeepShift](https://github.com/mostafaelhoushi/DeepShift) by Mostafa Elhoushi et
al.
