"""
Phase 1 — Train MLP on MNIST and export raw weights
=====================================================
Run this on YOUR machine (or Google Colab) where
MNIST can download freely.

Output files (saved to ../weights/):
  fc1_weight.txt   128 x 784  floats
  fc1_bias.txt     128        floats
  fc2_weight.txt   10  x 128  floats
  fc2_bias.txt     10         floats
  test_image.txt   784        floats  (one test sample)
  test_label.txt   1          int     (expected digit)
  accuracy.txt     training stats
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 0.001
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  DATASET  (downloads ~11 MB MNIST)
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("=" * 50)
print("  Phase 1 — Neural Digit Engine Training")
print("=" * 50)
print("  Downloading MNIST dataset...")

train_data   = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000,       shuffle=False)

print(f"  Train samples : {len(train_data)}")
print(f"  Test  samples : {len(test_data)}")

# ─────────────────────────────────────────
#  MODEL   784 → ReLU(128) → 10
# ─────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 128)
        self.fc2  = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)           # flatten 28x28 to 784
        x = self.relu(self.fc1(x))    # hidden layer + ReLU
        x = self.fc2(x)               # output logits (10)
        return x

model     = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
print("\n  Training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    print(f"  Epoch {epoch}/{EPOCHS}  |  Loss: {avg:.4f}")

# ─────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

accuracy = 100 * correct / total
print(f"\n  Test Accuracy : {accuracy:.2f}%")

# ─────────────────────────────────────────
#  HELPER — save tensor to .txt
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
    print(f"  Saved  {filename:25s}  shape={arr.shape}")

# ─────────────────────────────────────────
#  EXTRACT WEIGHTS
# ─────────────────────────────────────────
print("\n  Extracting weights to /weights/ ...")
save_weight(model.fc1.weight, 'fc1_weight.txt')   # (128, 784)
save_weight(model.fc1.bias,   'fc1_bias.txt')     # (128,)
save_weight(model.fc2.weight, 'fc2_weight.txt')   # (10, 128)
save_weight(model.fc2.bias,   'fc2_bias.txt')     # (10,)

# ─────────────────────────────────────────
#  SAVE ONE VERIFICATION IMAGE + LABEL
# ─────────────────────────────────────────
sample_img, sample_lbl = test_data[0]
pixels   = sample_img.view(-1).numpy()
expected = sample_lbl

with open(os.path.join(WEIGHTS_DIR, 'test_image.txt'), 'w') as f:
    for v in pixels:
        f.write(f"{v:.8f}\n")

with open(os.path.join(WEIGHTS_DIR, 'test_label.txt'), 'w') as f:
    f.write(str(expected) + '\n')

print(f"  Saved  test_image.txt             shape=(784,)")
print(f"  Saved  test_label.txt             expected digit = {expected}")

# ─────────────────────────────────────────
#  SAVE ACCURACY STATS
# ─────────────────────────────────────────
with open(os.path.join(WEIGHTS_DIR, 'accuracy.txt'), 'w') as f:
    f.write(f"Test Accuracy : {accuracy:.2f}%\n")
    f.write(f"Epochs        : {EPOCHS}\n")
    f.write(f"Batch Size    : {BATCH_SIZE}\n")
    f.write(f"Learning Rate : {LR}\n")
    f.write(f"Architecture  : 784 -> ReLU(128) -> 10\n")

print(f"  Saved  accuracy.txt")
print("\n  Phase 1 complete! All weight files ready.")
print("=" * 50)