import os
import csv
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imna
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from setup_database import CsvLogger

EPOCHS = 10
BATCH_SIZE = 2000
LR = 0.01
# PRETRAINED = "mnist_shift_rounded.pt"  # set to None to train from scratch
PRETRAINED = None
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BATCH_LIMIT_PER_EPOCH = 50

db = CsvLogger(RESULTS_DIR)
run_id    = db.create_run(notes=f"'mnist epochs={EPOCHS} lr={LR}, with re-usable dpu allocation, shuffle false'", batch_size=BATCH_SIZE)
_conv_csv = db.conv_csv_path()


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = accel.UpmemConv2d(1, 32, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_csv_file=_conv_csv)
#         self.conv2 = accel.UpmemConv2d(32, 64, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_csv_file=_conv_csv)
#         self.fc1 = nn.Linear(12544, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         t0 = time.perf_counter()
#         x = self.conv1(x)
#         self.t_conv1 = time.perf_counter() - t0
#         self.t_conv1_quantize = self.conv1.t_quantize
#         self.t_conv1_dpu      = self.conv1.t_dpu

#         x = F.relu(x)

#         t0 = time.perf_counter()
#         x = self.conv2(x)
#         self.t_conv2 = time.perf_counter() - t0
#         self.t_conv2_quantize = self.conv2.t_quantize
#         self.t_conv2_dpu      = self.conv2.t_dpu

#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         # x = self.dropout1(x)
#         x = torch.flatten(x, 1).float()
#         x = self.fc1(x)
#         x = F.relu(x)
#         # x = self.dropout2(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = imna.imna.UpmemConv2dTraining(1, 32, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_log_file="conv1_timing.csv")
            self.conv2 = imna.imna.UpmemConv2dTraining(32, 64, kernel_size=3, stride=1, padding=1, dpu_logs=False, dpu_limit=BATCH_SIZE, timing_log_file="conv2_timing.csv")
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="../data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

model = Net()

if PRETRAINED is not None:
    try:
        model.load_state_dict(torch.load(PRETRAINED, map_location="cpu"))
        print(f"Loaded pretrained weights from {PRETRAINED}")
    except FileNotFoundError:
        print(f"Warning: {PRETRAINED} not found — training from scratch")

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

def train(epoch):
    model.train()
    t_zero = t_fwd = t_loss = t_bwd = t_step = 0.0
    t_conv1 = t_conv1_q = t_conv1_dpu = 0.0
    t_conv2 = t_conv2_q = t_conv2_dpu = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
    for i, (images, labels) in enumerate(pbar):
        if i >= BATCH_LIMIT_PER_EPOCH:
            break
        batch_id = db.begin_batch(run_id, epoch=epoch, batch_idx=i)
        model.conv1.batch_id = batch_id
        model.conv2.batch_id = batch_id

        t0 = time.perf_counter()
        optimizer.zero_grad()
        t_zero = time.perf_counter() - t0

        t0 = time.perf_counter()
        output = model(images)
        t_fwd = time.perf_counter() - t0
        t_conv1     = model.t_conv1
        t_conv1_q   = model.t_conv1_quantize
        t_conv1_dpu = model.t_conv1_dpu
        t_conv2     = model.t_conv2
        t_conv2_q   = model.t_conv2_quantize
        t_conv2_dpu = model.t_conv2_dpu

        t0 = time.perf_counter()
        loss = F.nll_loss(output, labels)
        t_loss = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        t_bwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        t_step = time.perf_counter() - t0

        db.finish_batch(batch_id,
            zero_grad_ms=t_zero * 1e3,
            forward_ms=t_fwd * 1e3,
            conv1_ms=t_conv1 * 1e3,
            conv1_quantize_ms=t_conv1_q * 1e3,
            conv1_dpu_ms=t_conv1_dpu * 1e3,
            conv2_ms=t_conv2 * 1e3,
            conv2_quantize_ms=t_conv2_q * 1e3,
            conv2_dpu_ms=t_conv2_dpu * 1e3,
            loss_ms=t_loss * 1e3,
            backward_ms=t_bwd * 1e3,
            step_ms=t_step * 1e3,
        )

        pbar.set_postfix(loss=f"{loss.item():.4f}")


def evaluate(epoch):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            batch_id = db.begin_batch(run_id, epoch=epoch, batch_idx=i)
            model.conv1.batch_id = batch_id
            model.conv2.batch_id = batch_id

            t0 = time.perf_counter()
            preds = model(images).argmax(dim=1)
            t_fwd = time.perf_counter() - t0

            db.finish_batch(batch_id, forward_ms=t_fwd * 1e3)

            all_preds.append(preds)
            all_labels.append(labels)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean()
    print(f"Test accuracy: {accuracy.item() * 100:.2f}%")


for epoch in range(1, EPOCHS + 1):
    train(epoch)
    raise Exception("Stopping after one epoch for testing purposes")  # remove this line to train for all epochs
    evaluate(epoch)
    scheduler.step()

torch.save(model.state_dict(), "mnist_shift_rounded_2.pt")
print("Model saved to mnist_shift_rounded_2.pt")

