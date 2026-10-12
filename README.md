# IMNA UPMEM

Code from my [thesis]( https://purl.utwente.nl/essays/110096) on running multiplication-free (bit-shift) neural network
layers on the UPMEM processing-in-memory (PIM)
architecture. It contains DPU kernels for shift-based convolution, a PyTorch
bridge that offloads layers of a trained model to the DPUs, and the scripts and
measurements used for the evaluation.

## Repository layout

| Folder | What it contains |
| --- | --- |
| [`upmem_menke/`](upmem_menke) | The main DPU program: a governor/worker kernel that runs shift-based `conv2d`, ReLU and max-pooling layers on the DPUs, plus a standalone C host program. `upmem_ai` loads the DPU binary built here. |
| [`upmem_ai/`](upmem_ai) | `imna`, a PyTorch extension (C++ host bridge + Python wrappers) that transforms a trained shift network and runs its layers on the DPUs. Also contains the MNIST training/inference scripts, trained models (`*.pt`), the measurement scripts and their results (`measurements/`, `results/`). |
| [`upmem_menke_multi_impementation/`](upmem_menke_multi_impementation) | Seven alternative DPU convolution kernels (`implementationA.c` to `implementationG.c`: DeepShift optimized and baseline, naive int/float, AdderNet int/float, and a variant without alignment compensation) with a script that benchmarks them against each other (`measure.py`, results in `measurements.csv`). |
| [`SDK/`](SDK) | The UPMEM SDK 2025.1.0 packages used to build the dev container, plus a hello-world sanity check (`test_sdk.sh`). |
| [`.devcontainer/`](.devcontainer) | Dev container (Ubuntu 22.04, UPMEM SDK in simulator mode, Python 3.13, CUDA toolkit) used for all development. |

## Getting started

### 1. Open the dev container

The easiest way to get a working toolchain is the dev container. Open the
repository in VS Code with the Dev Containers extension (or any tool that
supports `devcontainer.json`) and reopen it in the container. The image is built
for `linux/amd64` and installs the UPMEM SDK from `SDK/2025.1.0/`. Every shell
sources `/usr/upmem_env.sh simulator`, so programs run on the functional
simulator unless real UPMEM hardware is present.

Check that the SDK works:

```bash
./SDK/test_sdk.sh
```

### 2. Build the DPU program

```bash
cd upmem_menke
make            # produces build/bin/dpu and build/bin/host
```

Compile-time parameters such as `NR_TASKLETS`, `KERNEL_SIZE`, `FEATURE_SIZE`,
`PADDING`, `STRIDE` and `DPU_COUNT` can be overridden on the `make` command line.

### 3. Build and use the PyTorch bridge

```bash
cd upmem_ai
python3 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
./install.sh    # builds the imna C++ extension in place
```

Use a virtual environment as shown. Installing into the system Python can fail
with `Cannot uninstall sympy ... distutils installed project`, because PyTorch
tries to upgrade the `sympy` that Ubuntu installed through apt.

`mnist2.py` runs inference on MNIST with the trained shift model offloaded to
the DPUs, and `measure.py` collects the timing measurements. MNIST is downloaded
to `data/` in the repository root on first use.

### 4. Compare the kernel implementations

```bash
cd upmem_menke_multi_impementation
make            # builds build/bin/dpu_a ... dpu_g
python measure.py
```

### Note on paths

The Python scripts find DPU binaries and result folders relative to their own
location, so the repository can be cloned anywhere. Two things still assume a
layout:

- The scripts in `upmem_ai/` download MNIST to `../data` and should be run from
  inside `upmem_ai/`.
- The `imna` bridge writes an MRAM dump to `../upmem_menke/dpu/mram_init.h`
  relative to the working directory. Build with `-DMRAM_INIT_PATH=...` to change it.

## Credits

- **DeepShift** by Mostafa Elhoushi, Zihao Chen, Farhan Shafiq, Ye Henry Tian
  and Joey Yiwei Li ([github.com/mostafaelhoushi/DeepShift](https://github.com/mostafaelhoushi/DeepShift)).
  The helper files `upmem_ai/imna/ste.py` and `upmem_ai/imna/utils.py` are
  copied from (and lightly adapted from) DeepShift's `pytorch/deepshift/`
  package, and the shift-based kernels follow the DeepShift approach.

  ```bibtex
  @InProceedings{Elhoushi_2021_CVPR,
      author    = {Elhoushi, Mostafa and Chen, Zihao and Shafiq, Farhan and Tian, Ye Henry and Li, Joey Yiwei},
      title     = {DeepShift: Towards Multiplication-Less Neural Networks},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month     = {June},
      year      = {2021},
      pages     = {2359-2368}
  }
  ```

- The UPMEM SDK in `SDK/` is distributed by [UPMEM](https://www.upmem.com/) under its own license.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{veerman2026imna,
  author = {Veerman, Menke},
  title  = {Design and Evaluation of Governor–Worker Mapping for DeepShift on UPMEM PIM},
  school = {University of Twente},
  year   = {2026},
  url    = {https://purl.utwente.nl/essays/110096}
}
```
