import os
import time

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import imna
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import imna.ste as ste

BATCH_SIZE = 1000
NUM_BATCHES = 5
LEGACY=False
if LEGACY:
    PRETRAINED = "mnist_shift_rounded.pt"
else:
    PRETRAINED = "mnist_shift_rounded_fused.pt"
DPU_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upmem_menke", "build", "bin", "dpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
test_loader  = DataLoader(
    Subset(test_dataset, range(BATCH_SIZE * NUM_BATCHES)),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

if not LEGACY:
    # PIM version
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.handle = imna.IMNABridge(dpu_limit=BATCH_SIZE, dpu_binary=DPU_BINARY)
            self.conv1 = imna.Conv2D(
                self.handle, 1, 32, kernel_size=3, padding=1,
                dpu_logs=False, applyReluAfter=True)
            self.conv2 = imna.Conv2D(
                self.handle, 32, 64, kernel_size=3, padding=1,
                dpu_logs=False, applyReluAfter=True)
            self.maxpool = imna.MaxPooling(
                self.handle, channels=64, kernel_size=2, stride=2)
            self.fc1 = nn.Linear(12544, 128)
            self.fc2 = nn.Linear(128, 10)
            self._times = {k: [] for k in ['quantize', 'conv1', 'conv2', 'maxpool', 'download', 'fc1', 'fc2']}

        def build(self, batch_size: int):
            self.handle.build(input_shape=(batch_size, 1, 28, 28))

        def forward(self, x: torch.Tensor):
            t = time.perf_counter()
            x_int = ste.round_fixed_point(x, 16, 16).to(torch.int32)
            self._times['quantize'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x_dpu = self.conv1(imna.DpuTensor(x_int))
            self._times['conv1'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x_dpu = self.conv2(x_dpu)
            self._times['conv2'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x_dpu = self.maxpool(x_dpu)
            self._times['maxpool'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x = self.handle.download(x_dpu)
            self._times['download'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x = torch.flatten(x, 1).float()
            x = self.fc1(x)
            x = F.relu(x)
            self._times['fc1'].append(time.perf_counter() - t)

            t = time.perf_counter()
            x = self.fc2(x)
            result = F.log_softmax(x, dim=1)
            self._times['fc2'].append(time.perf_counter() - t)

            return result

else:
    # Legacy version
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = imna.eager.Conv2D(1, 32, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_log_file="conv1_timing.csv")
            self.conv2 = imna.eager.Conv2D(32, 64, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_log_file="conv2_timing.csv")
            # self.dropout1 = nn.Dropout(0.25)
            # self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(12544, 128)
            self.fc2 = nn.Linear(128, 10)
            self.accel = imna.IMNABridge(dpu_limit=BATCH_SIZE)

        def build(self, batch_size: int):
            pass

        def forward(self, x):
            t0 = time.perf_counter()
            x = self.conv1(x, self.accel)
            self.t_conv1 = time.perf_counter() - t0
            self.t_conv1_quantize = self.conv1.t_quantize
            self.t_conv1_dpu      = self.conv1.t_dpu

            x = F.relu(x)

            t0 = time.perf_counter()
            x = self.conv2(x, self.accel)
            self.t_conv2 = time.perf_counter() - t0
            self.t_conv2_quantize = self.conv2.t_quantize
            self.t_conv2_dpu      = self.conv2.t_dpu

            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            # x = self.dropout1(x)
            print(x.dtype)
            x = torch.flatten(x, 1).float()
            x = self.fc1(x)
            x = F.relu(x)
            # x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

model = Net()
model.build(BATCH_SIZE)

print("Loading pretrained weights from", PRETRAINED)
state_dict = torch.load(PRETRAINED, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# print(f"conv1 MRAM layout: weight_offset={model.conv1.descriptor.weight_offset_bytes}, "
#       f"input_offset={model.conv1.descriptor.input_offset_bytes}, "
#       f"output_offset={model.conv1.descriptor.output_offset_bytes}")
# print(f"conv2 MRAM layout: weight_offset={model.conv2.descriptor.weight_offset_bytes}, "
#       f"input_offset={model.conv2.descriptor.input_offset_bytes}, "
#       f"output_offset={model.conv2.descriptor.output_offset_bytes}")

correct = 0
total = 0

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        t0 = time.perf_counter()
        output = model(images)
        t_s = time.perf_counter() - t0
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(**{"s/batch": f"{t_s:.3f}", "acc": f"{100*correct/total:.1f}%"})

print(f"\nFinal accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

if not LEGACY:
    keys = [k for k, v in model._times.items() if v]
    print("\n" + "\t".join(keys))
    print("\t".join(f"{sum(model._times[k])/len(model._times[k]):.4f}" for k in keys))
