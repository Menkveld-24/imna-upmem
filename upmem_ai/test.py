import sys
import torch
import imna

# --- Manually defined kernels ---
# Shape: (out_channels=2, in_channels=2, kH=3, kW=3)
# Shift values must be integers in [-14, 0] (for default weight_bits=5)

shift_kernel = torch.tensor([
    # Filter 0: channel 0, channel 1, channel 2
    [[[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]],
     [[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]],
     [[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]]],
    # Filter 1: channel 0, channel 1, channel 2
    [[[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]],
     [[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]],
     [[ 0,  1,  7],
      [ 3, -4,  9],
      [ 0, 13,  2]]],
], dtype=torch.float32)  # shape: (2, 2, 3, 3)

sign_kernel = torch.tensor([
    # Filter 0: channel 0, channel 1, channel 2
    [[[ 1,  1,  1],
      [ 1, -1,  1],
      [ 1,  1,  1]],
     [[ 1,  1,  1],
      [ 1, -1,  1],
      [ 1,  1,  1]],
     [[ 1,  1,  1],
      [ 1, -1,  1],
      [ 1,  1,  1]]],
    # Filter 1: channel 0, channel 1
    [[[ 0,  0,  0],
      [ 0,  0,  0],
      [ 0,  0,  0]],
     [[ 0,  0,  0],
      [ 0,  0,  0],
      [ 0,  0,  0]],
     [[ 0,  0,  0],
      [ 0,  0,  0],
      [ 0,  0,  0]]],
], dtype=torch.float32)  # shape: (2, 2, 3, 3)

# --- Build model and inject kernels ---
model = imna.UpmemConv2d(3, 2, 3, padding=1, dpu_logs=False)

with torch.no_grad():
    model.shift_fp.data = shift_kernel
    model.sign_fp.data = sign_kernel

# --- Input feature: 28x28, row i has value i (0-indexed) ---
# Shape: (batch=1, channels=1, H=28, W=28)
row_values = torch.arange(28, dtype=torch.float32)          # [0, 1, ..., 27]
x = row_values.view(28, 1).expand(28, 28)                   # (28, 28) each row filled with its index
x = x.unsqueeze(0).unsqueeze(0)                             # (1, 1, 28, 28)
x = x.expand(1, 3, -1, -1)                                  # (1, 2, 28, 28) — 1 batch, 2 identical input channels
# Add a second batch with all values = 0 to test batch processing
x = torch.cat([x, torch.zeros_like(x)], dim=0)              # (2, 3, 28, 28)

y = model(x)

def format_filter(tensor_2d):
    """Format a 2D tensor as space-separated integers, one row per line."""
    lines = []
    for row in tensor_2d:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)

file_text = ""
for batch in range(x.shape[0]):
    file_text += f"Batch {batch}:\n"
    for c in range(x.shape[1]):
        file_text += f"Input channel {c}:\n{format_filter(x[batch, c])}\n\n"

    for filt in range(shift_kernel.shape[0]):
        for c in range(shift_kernel.shape[1]):
            file_text += f"Shift kernel filter {filt} channel {c}:\n{format_filter(shift_kernel[filt, c])}\n\n"
            file_text += f"Sign kernel filter {filt} channel {c}:\n{format_filter(sign_kernel[filt, c])}\n\n"

    for filt in range(y.shape[1]):
        file_text += f"Output filter {filt}:\n{format_filter(y[batch, filt])}\n\n"

with open("output.txt", "w") as f:
    f.write(file_text)
