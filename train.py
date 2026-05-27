"""
Phase 4 — train.py  (CNN upgrade)
================================================================
Architecture:
  Input  : 1 × 28 × 28
  Conv1  : 32 filters, 3×3, padding=1  → 32 × 28 × 28  + ReLU + BN
  Conv2  : 64 filters, 3×3, padding=1  → 64 × 28 × 28  + ReLU + BN
  Pool   : MaxPool 2×2                  → 64 × 14 × 14
  Dropout: 0.25
  Flatten: 64 × 7 × 7 = 3136
  FC1    : 12544 → 128                  + ReLU + BN + Dropout(0.5)
  FC2    : 128   → 10                   (logits)

Same augmentation as Phase 3.
Same weight export format (.txt) — engine.cpp is rewritten for CNN.

Exported files (weights/):
  conv1_weight.txt   32 × 1 × 3 × 3  (each filter on one line, 9 values)
  conv1_bias.txt     32
  conv2_weight.txt   64 × 32 × 3 × 3
  conv2_bias.txt     64
  fc1_weight.txt     128 × 3136
  fc1_bias.txt       128
  fc2_weight.txt     10  × 128
  fc2_bias.txt       10

Run:  python train.py
Then: cd cpp && g++ -O2 -std=c++17 -o engine engine.cpp
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
EPOCHS       = 15
BATCH_SIZE   = 64    # smaller batch = less RAM per step
LR           = 0.001
PATIENCE     = 5
LABEL_SMOOTH = 0.1

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# ─────────────────────────────────────────
# AUGMENTATION  (same as Phase 3)
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,
        translate=(0.10, 0.10),
        scale=(0.85, 1.15),
        shear=5,
    ),
    transforms.ElasticTransform(alpha=20.0, sigma=5.0),
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])

print("=" * 55)
print("  Phase 4 — CNN Training")
print("  Conv(32) → Conv(64) → Pool → FC(128) → 10")
print("=" * 55)
print("  Loading MNIST...")

train_data   = datasets.MNIST('./data', train=True,  download=True, transform=train_transform)
test_data    = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=1000,        shuffle=False, num_workers=0)

print(f"  Train: {len(train_data)}  Test: {len(test_data)}\n")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → 32×28×28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 32×14×14
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×14×14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 64×7×7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),                   # 3136 → 128 (4x smaller)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model     = CNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print(f"  Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

best_acc      = 0.0
patience_left = PATIENCE
best_state    = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100 * correct / total

    lr_now = scheduler.get_last_lr()[0]
    marker = ''
    if acc > best_acc:
        best_acc      = acc
        patience_left = PATIENCE
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        marker = '  ← best'
    else:
        patience_left -= 1

    scheduler.step()
    print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  "
          f"acc={acc:.2f}%  lr={lr_now:.6f}{marker}")

    if patience_left == 0:
        print(f"\n  Early stopping at epoch {epoch}")
        break

print(f"\n  Best test accuracy: {best_acc:.2f}%")

# ─────────────────────────────────────────
# PER-DIGIT REPORT
# ─────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
print("\n  Per-digit accuracy:")

class_correct = [0] * 10
class_total   = [0] * 10
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        for label, pred in zip(labels, preds):
            class_total[label]   += 1
            class_correct[label] += int(pred == label)

for d in range(10):
    dacc = 100 * class_correct[d] / class_total[d]
    bar  = '█' * int(dacc / 2)
    flag = '  ← weakest' if d == min(range(10), key=lambda i: class_correct[i]/class_total[i]) else ''
    print(f"    {d}: {dacc:5.2f}%  {bar}{flag}")

# ─────────────────────────────────────────
# EXPORT WEIGHTS
# BN is folded into preceding Conv/Linear
# so C++ engine needs no BN logic.
# ─────────────────────────────────────────
def fold_bn_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold BN into Conv2d. Returns (W_eff, b_eff)."""
    W      = conv.weight.detach()        # (out, in, kH, kW)
    b      = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
    gamma  = bn.weight.detach()
    beta   = bn.bias.detach()
    mean   = bn.running_mean.detach()
    var    = bn.running_var.detach()
    std    = (var + bn.eps).sqrt()
    scale  = gamma / std                 # (out,)
    W_eff  = W * scale.view(-1, 1, 1, 1)
    b_eff  = scale * (b - mean) + beta
    return W_eff, b_eff

def fold_bn_linear(linear: nn.Linear, bn: nn.BatchNorm1d):
    """Fold BN into Linear. Returns (W_eff, b_eff)."""
    W      = linear.weight.detach()
    b      = linear.bias.detach()
    gamma  = bn.weight.detach()
    beta   = bn.bias.detach()
    mean   = bn.running_mean.detach()
    var    = bn.running_var.detach()
    std    = (var + bn.eps).sqrt()
    scale  = gamma / std
    W_eff  = W * scale.unsqueeze(1)
    b_eff  = scale * (b - mean) + beta
    return W_eff, b_eff

def save_flat(tensor, filename):
    """Save any tensor as a flat list of floats, one per line."""
    arr  = tensor.detach().numpy().flatten()
    path = os.path.join(WEIGHTS_DIR, filename)
    with open(path, 'w') as f:
        for v in arr:
            f.write(f"{v:.8f}\n")
    print(f"  Saved {filename:28s}  elements={len(arr):>8,}  shape={tuple(tensor.shape)}")

print("\n  Exporting weights (BN folded)...")

# features: [Conv2d, BN2d, ReLU, MaxPool2d, Conv2d, BN2d, ReLU, MaxPool2d, Dropout2d]
#            0       1     2     3          4       5     6     7            8
conv1_W, conv1_b = fold_bn_conv(model.features[0], model.features[1])
conv2_W, conv2_b = fold_bn_conv(model.features[4], model.features[5])

# classifier: [Linear, BN1d, ReLU, Dropout, Linear]
#              0       1     2     3        4
fc1_W, fc1_b = fold_bn_linear(model.classifier[0], model.classifier[1])
fc2_W = model.classifier[4].weight.detach()
fc2_b = model.classifier[4].bias.detach()

save_flat(conv1_W, 'conv1_weight.txt')   # 32×1×3×3  = 288 values
save_flat(conv1_b, 'conv1_bias.txt')     # 32
save_flat(conv2_W, 'conv2_weight.txt')   # 64×32×3×3 = 18432 values
save_flat(conv2_b, 'conv2_bias.txt')     # 64
save_flat(fc1_W,   'fc1_weight.txt')     # 128×12544
save_flat(fc1_b,   'fc1_bias.txt')       # 128
save_flat(fc2_W,   'fc2_weight.txt')     # 10×128
save_flat(fc2_b,   'fc2_bias.txt')       # 10

# remove old MLP-only files so engine doesn't accidentally load them
for old in ['fc3_weight.txt', 'fc3_bias.txt']:
    p = os.path.join(WEIGHTS_DIR, old)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Removed old {old}")

# save checkpoint + verification sample
torch.save(best_state, os.path.join(WEIGHTS_DIR, 'best_model_cnn.pt'))
sample_img, sample_lbl = test_data[0]
pixels = sample_img.view(-1).numpy()
with open(os.path.join(WEIGHTS_DIR, 'test_image.txt'), 'w') as f:
    for v in pixels:
        f.write(f"{v:.8f}\n")
with open(os.path.join(WEIGHTS_DIR, 'test_label.txt'), 'w') as f:
    f.write(str(sample_lbl) + '\n')

with open(os.path.join(WEIGHTS_DIR, 'accuracy.txt'), 'w') as f:
    f.write(f"Phase         : 4 (CNN)\n")
    f.write(f"Test Accuracy : {best_acc:.2f}%\n")
    f.write(f"Architecture  : Conv(32) → Conv(64) → MaxPool → FC(128) → 10\n")
    f.write(f"Parameters    : {total_params:,}\n")
    f.write("\nPer-digit accuracy:\n")
    for d in range(10):
        dacc = 100 * class_correct[d] / class_total[d]
        f.write(f"  digit {d}: {dacc:.2f}%\n")

print("\n  Phase 4 CNN training complete!")
print(f"  Best accuracy: {best_acc:.2f}%")
print("=" * 55)