"""
Transform a .pt model file for DPU inference.
Fuses sign+shift weight pairs into the fused_ss format expected by UpmemConv2d.

Usage: python transform_model.py <model.pt> [output.pt]

If output.pt is omitted, saves alongside the input with a _fused suffix,
e.g. mnist.pt -> mnist_fused.pt.
"""

import os
import sys
import torch
from imna import ModelTransformer


def _load_state_dict(path: str) -> tuple[dict, dict | None]:
    """Load a checkpoint and return (state_dict, full_data).
    full_data is None when the file is a bare state_dict.
    """
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"], data
    elif isinstance(data, dict):
        return data, None
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(data)}")


def inspect(state_dict: dict, path: str) -> None:
    print(f"File : {path}")
    print(f"Keys : {len(state_dict)}\n")

    col_w = max(len(k) for k in state_dict) + 2
    header = (
        f"{'Key':<{col_w}} {'Shape':<24} {'Dtype':<12}"
        f" {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}"
    )
    print(header)
    print("-" * len(header))

    for key, tensor in state_dict.items():
        t = tensor.float()
        print(
            f"{key:<{col_w}} {str(tuple(tensor.shape)):<24} {str(tensor.dtype):<12}"
            f" {t.min().item():>12.6f} {t.max().item():>12.6f}"
            f" {t.mean().item():>12.6f} {t.std().item():>12.6f}"
        )

    print()
    print(f"Total parameters: {sum(t.numel() for t in state_dict.values()):,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_model.py <model.pt> [output.pt]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else (
        os.path.splitext(input_path)[0] + "_fused" + (os.path.splitext(input_path)[1] or ".pt")
    )

    sd, full_data = _load_state_dict(input_path)
    inspect(sd, input_path)

    print("\nTransforming weights...")
    out_sd = ModelTransformer(sd).to_fused_ss().get()

    if full_data is not None:
        full_data["state_dict"] = out_sd
        torch.save(full_data, output_path)
    else:
        torch.save(out_sd, output_path)

    print(f"\nSaved to: {output_path}")
