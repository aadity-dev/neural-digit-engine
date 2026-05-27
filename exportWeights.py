"""
export_weights.py — run this after training to re-export weights only.
Loads best_model_cnn.pt and saves all weight .txt files.
Usage: python export_weights.py
"""
import torch, torch.nn as nn, os

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

model = CNN()
ckpt  = os.path.join(WEIGHTS_DIR, 'best_model_cnn.pt')
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()
print(f"Loaded {ckpt}")

def fold_bn_conv(conv, bn):
    W = conv.weight.detach(); b = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
    s = bn.weight.detach() / (bn.running_var.detach() + bn.eps).sqrt()
    return W * s.view(-1,1,1,1), s*(b - bn.running_mean.detach()) + bn.bias.detach()

def fold_bn_linear(lin, bn):
    W = lin.weight.detach(); b = lin.bias.detach()
    s = bn.weight.detach() / (bn.running_var.detach() + bn.eps).sqrt()
    return W * s.unsqueeze(1), s*(b - bn.running_mean.detach()) + bn.bias.detach()

def save_flat(t, name):
    arr = t.detach().numpy().flatten()
    with open(os.path.join(WEIGHTS_DIR, name), 'w') as f:
        for v in arr: f.write(f"{v:.8f}\n")
    print(f"  Saved {name:28s} ({len(arr):,} values)")

# features: 0=Conv,1=BN,2=ReLU,3=Pool, 4=Conv,5=BN,6=ReLU,7=Pool,8=Drop
conv1_W, conv1_b = fold_bn_conv(model.features[0], model.features[1])
conv2_W, conv2_b = fold_bn_conv(model.features[4], model.features[5])
fc1_W,   fc1_b   = fold_bn_linear(model.classifier[0], model.classifier[1])
fc2_W = model.classifier[4].weight.detach()
fc2_b = model.classifier[4].bias.detach()

save_flat(conv1_W, 'conv1_weight.txt')
save_flat(conv1_b, 'conv1_bias.txt')
save_flat(conv2_W, 'conv2_weight.txt')
save_flat(conv2_b, 'conv2_bias.txt')
save_flat(fc1_W,   'fc1_weight.txt')
save_flat(fc1_b,   'fc1_bias.txt')
save_flat(fc2_W,   'fc2_weight.txt')
save_flat(fc2_b,   'fc2_bias.txt')
print("\nWeights exported! Now recompile: cd cpp && g++ -O2 -std=c++17 -o engine engine.cpp")
