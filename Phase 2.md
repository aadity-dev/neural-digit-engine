# Phase 2 — Setup & Migration Guide

## What changed and why

| File | What changed | Why |
|---|---|---|
| `train.py` | Removed `print/exit()` at top | Training never ran in Phase 1 |
| `train.py` | 784→256→128→10 + dropout + augmentation | Better accuracy on drawn digits |
| `backend/preprocess.py` | **New file** | Core fix for wrong predictions |
| `backend/app.py` | Uses preprocess, returns top-3 + all scores | Richer API response |
| `cpp/engine.cpp` | 3-layer forward pass + softmax | Correct probability output |
| `frontend/index.html` | Confidence bar chart + debug view | See what the engine sees |
| `.gitignore` | Added `*.dSYM/`, `cpp/engine` | Clean repo |
| `requirements.txt` | Added `Pillow`, `flask-cors` | Preprocessing needs PIL |

---

## Step 1 — Retrain the model

```bash
# From repo root
pip install -r requirements.txt
python train.py
```

This now actually runs (Phase 1 had `exit()` at line 2). It will:
- Download MNIST (~11 MB) into `data/`
- Train for 15 epochs with augmentation
- Save **6 weight files** to `weights/` (3 layers × weight+bias)

Expected output: test accuracy ~98.5%

---

## Step 2 — Recompile the C++ engine

The engine is now a 3-layer network (Phase 1 was 2-layer).
You **must** recompile, otherwise it will read the wrong weight shapes.

```bash
cd cpp
g++ -O2 -std=c++17 -o engine engine.cpp
```

Test it works:
```bash
cat ../weights/test_image.txt | ./engine
# Expected output:
# PREDICTION 7
# SCORES 0.00001 0.00002 ... 0.99 ...
```

---

## Step 3 — Run the backend

```bash
cd backend
python app.py
# Listening on http://localhost:5000
```

---

## Step 4 — Open the frontend

Open `frontend/index.html` in a browser (double-click or `open frontend/index.html`).

Draw a digit → click **Predict** → see:
- The predicted digit + confidence %
- A bar chart for all 10 digits
- A 28×28 debug image showing exactly what the engine processed

---

## Why predictions were wrong before (root cause)

The canvas sends **black ink on white background**.
MNIST trained on **white digits on black background**.

Without inversion, the model sees a "white page with a hole" — something it has never trained on. That's why 1, 4, 7 were predicted as 0 or 8.

`preprocess.py` fixes this with a 3-step pipeline:
1. **Invert** the image (255 - pixel)
2. **Crop** to bounding box of the digit + 10% padding
3. **Normalise** as `(pixel/255 - 0.1307) / 0.3081` — matching exactly what `train.py` does

---

## Troubleshooting

**`ENGINE ERROR: Cannot find weights/`**
→ Run `python train.py` from the repo root first.

**`ENGINE ERROR: Expected 784 pixels, got N`**
→ Preprocessing bug. Check `preprocess.py` is in `backend/` and imported correctly.

**`ModuleNotFoundError: PIL`**
→ Run `pip install Pillow`.

**CORS error in browser console**
→ Make sure `flask-cors` is installed and Flask is running on port 5000.