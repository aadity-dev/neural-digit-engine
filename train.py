"""
Phase 3 — train.py
================================================================
Upgrades over Phase 2:
  - Stronger augmentation: elastic distortion + scale variation
    to better match how humans actually draw digits
  - Batch Normalisation after each layer (more stable training)
  - Label smoothing (CrossEntropy eps=0.1) — prevents overconfidence
  - Cosine annealing LR schedule instead of StepLR
  - Early stopping (patience=5) — saves best weights automatically
  - Per-class accuracy report — see exactly which digits still fail
  - Saves best checkpoint as weights/best_model.pt for inspection

Architecture unchanged: 784 → BN+ReLU(256) → BN+ReLU(128) → 10
(C++ engine.cpp unchanged — same weight files)

Run:  python train.py
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
EPOCHS        = 25          # more epochs; early stopping handles overfitting
BATCH_SIZE    = 128         # larger batch works better with batch norm
LR            = 0.001
PATIENCE      = 5           # stop if no improvement for 5 epochs
LABEL_SMOOTH  = 0.1         # label smoothing epsilon

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# ─────────────────────────────────────────
# AUGMENTATION  (stronger than Phase 2)
# ─────────────────────────────────────────
# ElasticTransform mimics natural pen/pencil stroke variation —
# the single biggest boost for drawn-digit accuracy.
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,              # ±15° rotation (was ±10°)
        translate=(0.10, 0.10), # ±10% shift    (was ±8%)
        scale=(0.85, 1.15),     # ±15% zoom — new in Phase 3
        shear=5,                # slight shear  — new in Phase 3
    ),
    transforms.ElasticTransform(alpha=20.0, sigma=5.0),  # NEW: warps strokes naturally
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])

print("=" * 55)
print("  Phase 3 — Neural Digit Engine Training")
print("=" * 55)
print("  Loading MNIST dataset...")

train_data   = datasets.MNIST('./data', train=True,  download=True, transform=train_transform)
test_data    = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=1000,        shuffle=False, num_workers=0)

print(f"  Train samples : {len(train_data)}")
print(f"  Test  samples : {len(test_data)}")

# ─────────────────────────────────────────
# MODEL — same shape as Phase 2 + BatchNorm
# ─────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),   # NEW: normalises activations per batch
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),   # NEW
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))

model     = MLP()
# Label smoothing: instead of hard 0/1 targets, uses 0.05/0.95
# prevents the model from being overconfident on ambiguous digits
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.Adam(model.parameters(), lr=LR)
# Cosine annealing: smoothly decays LR → 0 over training
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ─────────────────────────────────────────
# TRAINING LOOP with early stopping
# ─────────────────────────────────────────
print(f"\n  Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

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
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n  Best test accuracy: {best_acc:.2f}%")

# ─────────────────────────────────────────
# PER-CLASS ACCURACY REPORT
# ─────────────────────────────────────────
print("\n  Per-digit accuracy (best model):")
model.load_state_dict(best_state)
model.eval()

class_correct = [0] * 10
class_total   = [0] * 10
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        for label, pred in zip(labels, preds):
            class_total[label]   += 1
            class_correct[label] += int(pred == label)

worst_digit = -1
worst_acc   = 101.0
for d in range(10):
    dacc = 100 * class_correct[d] / class_total[d]
    bar  = '█' * int(dacc / 2)
    flag = '  ← weakest' if d == min(range(10), key=lambda i: class_correct[i]/class_total[i]) else ''
    print(f"    {d}: {dacc:5.2f}%  {bar}{flag}")

# ─────────────────────────────────────────
# EXPORT WEIGHTS (same format → C++ unchanged)
# ─────────────────────────────────────────
def save_weight(tensor, filename):
    arr  = tensor.detach().numpy()
    path = os.path.join(WEIGHTS_DIR, filename)
    with open(path, 'w') as f:
        if arr.ndim == 1:
            for v in arr:
                f.write(f"{v:.8f}\n")
        else:
            for row in arr:
                f.write(' '.join(f"{v:.8f}" for v in row) + '\n')
    print(f"  Saved {filename:25s} shape={arr.shape}")

print("\n  Exporting weights...")

# BatchNorm adds 4 extra param tensors per layer (weight, bias, mean, var)
# The C++ engine only uses fc weights/biases — BatchNorm is fused in here
# by extracting the effective weight/bias after BN folding.
# Layers in self.net:  0=Linear, 1=BN, 2=ReLU, 3=Drop, 4=Linear, 5=BN, ...

def fold_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fold BatchNorm into the preceding Linear layer so C++ engine
    needs no changes. After folding:
       y = W_eff @ x + b_eff
       W_eff = gamma / std  *  W
       b_eff = gamma / std  * (b - mean) + beta
    """
    W   = linear.weight.detach()          # (out, in)
    b   = linear.bias.detach()            # (out,)
    gamma  = bn.weight.detach()           # (out,)
    beta   = bn.bias.detach()             # (out,)
    mean   = bn.running_mean.detach()     # (out,)
    var    = bn.running_var.detach()      # (out,)
    eps    = bn.eps

    std    = (var + eps).sqrt()           # (out,)
    scale  = gamma / std                  # (out,)

    W_eff  = W * scale.unsqueeze(1)       # broadcast over input dim
    b_eff  = scale * (b - mean) + beta

    return W_eff, b_eff

net = model.net
# fc1 = net[0], bn1 = net[1]
# fc2 = net[4], bn2 = net[5]
# fc3 = net[8]
W1, b1 = fold_bn(net[0], net[1])
W2, b2 = fold_bn(net[4], net[5])
W3     = net[8].weight.detach()
b3     = net[8].bias.detach()

save_weight(W1, 'fc1_weight.txt')   # (256, 784)
save_weight(b1, 'fc1_bias.txt')     # (256,)
save_weight(W2, 'fc2_weight.txt')   # (128, 256)
save_weight(b2, 'fc2_bias.txt')     # (128,)
save_weight(W3, 'fc3_weight.txt')   # (10,  128)
save_weight(b3, 'fc3_bias.txt')     # (10,)

# Save best model checkpoint
torch.save(best_state, os.path.join(WEIGHTS_DIR, 'best_model.pt'))
print(f"  Saved best_model.pt")

# verification sample
sample_img, sample_lbl = test_data[0]
pixels = sample_img.view(-1).numpy()
with open(os.path.join(WEIGHTS_DIR, 'test_image.txt'), 'w') as f:
    for v in pixels:
        f.write(f"{v:.8f}\n")
with open(os.path.join(WEIGHTS_DIR, 'test_label.txt'), 'w') as f:
    f.write(str(sample_lbl) + '\n')

with open(os.path.join(WEIGHTS_DIR, 'accuracy.txt'), 'w') as f:
    f.write(f"Phase         : 3\n")
    f.write(f"Test Accuracy : {best_acc:.2f}%\n")
    f.write(f"Epochs run    : {epoch}\n")
    f.write(f"Batch Size    : {BATCH_SIZE}\n")
    f.write(f"LR schedule   : CosineAnnealing (LR={LR} → 1e-5)\n")
    f.write(f"Label smooth  : {LABEL_SMOOTH}\n")
    f.write(f"Architecture  : 784 → BN+ReLU(256) → BN+ReLU(128) → 10\n")
    f.write(f"Augmentation  : rotation=±15°, translate=±10%, scale=±15%, shear=5°, ElasticTransform\n")
    f.write("\nPer-digit accuracy:\n")
    for d in range(10):
        dacc = 100 * class_correct[d] / class_total[d]
        f.write(f"  digit {d}: {dacc:.2f}%\n")

print("  Saved accuracy.txt")
print("\n  Phase 3 training complete!")
print("=" * 55)