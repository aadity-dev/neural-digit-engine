"""
Phase 5 — export_binary.py
===========================
Converts the .txt weight files to compact binary (.bin) format.
Each .bin file is raw float32 bytes — no headers, no newlines.

  txt size  ≈ 9 bytes/value  (e.g. "-0.12345678\n")
  bin size  =  4 bytes/value  (IEEE 754 float32)

Speed improvement in C++: fread() one call vs getline() per value.
  txt load time: ~200ms
  bin load time: ~5ms

Output files (weights/):
  conv1_weight.bin   288     floats = 1.1 KB
  conv1_bias.bin     32      floats = 0.1 KB
  conv2_weight.bin   18432   floats = 72  KB
  conv2_bias.bin     64      floats = 0.3 KB
  fc1_weight.bin     401408  floats = 1.5 MB
  fc1_bias.bin       128     floats = 0.5 KB
  fc2_weight.bin     1280    floats = 5   KB
  fc2_bias.bin       10      floats = 0.04 KB

Usage:
  python export_binary.py          # converts existing .txt → .bin
  python export_binary.py --verify # also verifies values match
"""

import struct
import os
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

WEIGHT_FILES = [
    ('conv1_weight.txt', 32 * 1 * 9),
    ('conv1_bias.txt',   32),
    ('conv2_weight.txt', 64 * 32 * 9),
    ('conv2_bias.txt',   64),
    ('fc1_weight.txt',   128 * 3136),
    ('fc1_bias.txt',     128),
    ('fc2_weight.txt',   10 * 128),
    ('fc2_bias.txt',     10),
]

verify = '--verify' in sys.argv

print("=" * 50)
print("  Phase 5 — Binary Weight Export")
print("=" * 50)

total_txt = 0
total_bin = 0

for txt_name, expected_count in WEIGHT_FILES:
    txt_path = os.path.join(WEIGHTS_DIR, txt_name)
    bin_name = txt_name.replace('.txt', '.bin')
    bin_path = os.path.join(WEIGHTS_DIR, bin_name)

    if not os.path.exists(txt_path):
        print(f"  SKIP {txt_name} — not found (run exportWeights.py first)")
        continue

    # Read floats from .txt
    with open(txt_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]

    if len(values) != expected_count:
        print(f"  ERROR {txt_name}: expected {expected_count}, got {len(values)}")
        continue

    # Write as raw float32 binary
    with open(bin_path, 'wb') as f:
        f.write(struct.pack(f'{len(values)}f', *values))

    txt_size = os.path.getsize(txt_path)
    bin_size = os.path.getsize(bin_path)
    ratio    = txt_size / bin_size

    total_txt += txt_size
    total_bin += bin_size

    # Optional verify — read back and compare
    if verify:
        with open(bin_path, 'rb') as f:
            readback = list(struct.unpack(f'{len(values)}f', f.read()))
        max_diff = max(abs(a - b) for a, b in zip(values, readback))
        ok = '✓' if max_diff < 1e-6 else '✗'
        print(f"  {ok} {bin_name:28s}  {bin_size/1024:7.1f} KB  (verify diff={max_diff:.2e})")
    else:
        print(f"  ✓ {bin_name:28s}  {bin_size/1024:7.1f} KB  ({ratio:.1f}x smaller than .txt)")

print(f"\n  Total .txt size : {total_txt/1024:.1f} KB")
print(f"  Total .bin size : {total_bin/1024:.1f} KB")
print(f"  Size reduction  : {total_txt/total_bin:.1f}x")
print("\n  Done! Now recompile: cd cpp && g++ -O2 -std=c++17 -o engine engine.cpp")
print("=" * 50)