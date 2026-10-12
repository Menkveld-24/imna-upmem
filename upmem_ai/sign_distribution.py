import torch

MODEL_PATH = 'mnist_shift_rounded_int8.pt'

sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']

sign_keys = [k for k in sd if k.endswith('.sign')]

all_signs = torch.cat([sd[k].flatten().long() for k in sign_keys])

values, counts = all_signs.unique(return_counts=True)
total = all_signs.numel()

print(f"Model: {MODEL_PATH}")
print(f"Total sign weights: {total}\n")

print(f"{'Value':>8}  {'Count':>8}  {'%':>8}")
print("-" * 30)
for v, c in zip(values.tolist(), counts.tolist()):
    print(f"{v:>8}  {c:>8}  {c/total*100:>7.2f}%")

print("\nPer layer:")
for k in sign_keys:
    t = sd[k].flatten().long()
    n = t.numel()
    vs, cs = t.unique(return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vs, cs)}
    parts = "  ".join(f"{v}: {dist.get(v,0):>5} ({dist.get(v,0)/n*100:.1f}%)" for v in [-1, 0, 1])
    print(f"  {k:<20} {parts}")
