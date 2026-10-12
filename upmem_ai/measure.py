import os
import csv
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imna
import imna.ste as ste
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from setup_database import CsvLogger


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--num-batches", type=int, default=5)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_BATCHES = args.num_batches
PRETRAINED = "mnist_shift_rounded_fused.pt"
DPU_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upmem_menke", "build", "bin", "dpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

from torch.utils.data import ConcatDataset
_test  = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
# _train = datasets.MNIST(root="../data", train=True,  download=True, transform=transform)
_full  = ConcatDataset([_test, _test, _test, _test])  # 70 000 images
test_loader = DataLoader(
    Subset(_full, range(BATCH_SIZE * NUM_BATCHES)),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

db = CsvLogger(RESULTS_DIR)
run_id    = db.create_run(notes=f"'mnist measurement'", batch_size=BATCH_SIZE)
_conv_csv = db.conv_csv_path()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        self._times = {k: [] for k in ['quantize', 'conv1', 'conv2', 'maxpool', 'download', 'fc']}

    def build(self, batch_size: int):
            self.handle.build(input_shape=(batch_size, 1, 28, 28))

    def forward(self, x):
        t = time.perf_counter()
        x = ste.round_fixed_point(x, 16, 16).to(torch.int32)
        self._times['quantize'].append(time.perf_counter() - t)
        
        t = time.perf_counter()
        x = self.conv1(imna.DpuTensor(x))
        self._times['conv1'].append(time.perf_counter() - t)

        t = time.perf_counter()
        x = self.conv2(x)
        self._times['conv2'].append(time.perf_counter() - t)

        t = time.perf_counter()
        x = self.maxpool(x)
        self._times['maxpool'].append(time.perf_counter() - t)

        t = time.perf_counter()
        x = self.handle.download(x)
        self._times['download'].append(time.perf_counter() - t)

        t = time.perf_counter()
        x = torch.flatten(x, 1).float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        result = F.log_softmax(x, dim=1)
        self._times['fc'].append(time.perf_counter() - t)

        return result

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

model = Net()
model.build(BATCH_SIZE)

print("Loading pretrained weights from", PRETRAINED)
state_dict = torch.load(PRETRAINED, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        batch_id = db.begin_batch(run_id, epoch=0, batch_idx=batch_idx)
        model.conv1.batch_id = batch_id
        model.conv2.batch_id = batch_id

        t0 = time.perf_counter()
        output = model(images)
        t_s = time.perf_counter() - t0
        db.finish_batch(
            batch_id,
            forward_ms=t_s * 1e3,
            conv1_ms=model._times['conv1'][-1] * 1e3,
            conv2_ms=model._times['conv2'][-1] * 1e3,
            maxpool_ms=model._times['maxpool'][-1] * 1e3,
            fc_ms=model._times['fc'][-1] * 1e3,
            conv1_quantize_ms=model.conv1.t_quantize * 1e3,
            conv1_dpu_ms=model.conv1.t_dpu * 1e3,
            conv2_quantize_ms=model.conv2.t_quantize * 1e3,
            conv2_dpu_ms=model.conv2.t_dpu * 1e3,
        )

        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(**{"s/batch": f"{t_s:.3f}", "acc": f"{100*correct/total:.1f}%"})

accuracy = 100 * correct / total
print(f"\nFinal accuracy: {correct}/{total} ({accuracy:.1f}%)")

keys = [k for k, v in model._times.items() if v]
print("\n" + "\t".join(keys))
print("\t".join(f"{sum(model._times[k])/len(model._times[k]):.4f}" for k in keys))

# Write summary row
_SUMMARY_HEADER = ["run_id", "batch_size", "num_batches", "correct", "total", "accuracy_pct"] + [f"{k}_avg_s" for k in keys]
_summary_row = [run_id, BATCH_SIZE, NUM_BATCHES, correct, total, f"{accuracy:.2f}"] + [
    f"{sum(model._times[k])/len(model._times[k]):.6f}" for k in keys
]
_summary_path = os.path.join(RESULTS_DIR, "summary.csv")
_write_header = not os.path.exists(_summary_path)
with open(_summary_path, "a", newline="") as _f:
    _w = csv.writer(_f)
    if _write_header:
        _w.writerow(_SUMMARY_HEADER)
    _w.writerow(_summary_row)
print(f"Summary written to {_summary_path}")
