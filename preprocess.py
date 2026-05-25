"""
preprocess.py  — shared preprocessing for canvas drawings
==========================================================
This is the KEY fix for wrong predictions.

The pipeline mirrors exactly what MNIST does:
  1. Decode base64 PNG from the browser canvas
  2. Convert to greyscale
  3. INVERT  (canvas = black ink on white; MNIST = white digit on black)
  4. Crop to the bounding box of the digit + padding
  5. Resize to 28 × 28
  6. Normalise: pixel = (pixel/255 - 0.1307) / 0.3081   ← must match train.py
  7. Return flat list of 784 floats ready for the C++ engine

Usage:
    from preprocess import canvas_to_mnist
    pixels_784 = canvas_to_mnist(base64_png_string)
"""

import base64
import io
import numpy as np
from PIL import Image, ImageOps

# MNIST normalisation constants — must match train.py exactly
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def canvas_to_mnist(b64_data: str) -> list[float]:
    """
    Convert a base64-encoded canvas PNG to a flat list of 784 normalised floats.
    Raises ValueError if image cannot be decoded.
    """
    # 1. Decode base64 → PIL image
    if ',' in b64_data:
        b64_data = b64_data.split(',', 1)[1]   # strip "data:image/png;base64,"
    try:
        img_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')   # greyscale
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")

    img_arr = np.array(img, dtype=np.float32)   # shape (H, W), values 0–255

    # 2. INVERT: canvas has dark ink on light background; MNIST is the opposite
    img_arr = 255.0 - img_arr

    # 3. Threshold to remove grey anti-aliasing noise
    img_arr = np.where(img_arr > 30, img_arr, 0.0)

    # 4. Bounding-box crop + padding
    #    Find rows/cols that contain non-zero pixels
    rows = np.any(img_arr > 0, axis=1)
    cols = np.any(img_arr > 0, axis=0)

    if not rows.any():
        # blank canvas — return all zeros
        return [0.0] * 784

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # add 10% padding around the digit
    h = rmax - rmin
    w = cmax - cmin
    pad_r = max(4, int(h * 0.1))
    pad_c = max(4, int(w * 0.1))

    rmin = max(0, rmin - pad_r)
    rmax = min(img_arr.shape[0] - 1, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(img_arr.shape[1] - 1, cmax + pad_c)

    cropped = img_arr[rmin:rmax+1, cmin:cmax+1]

    # 5. Resize to 28 × 28 using LANCZOS
    pil_crop = Image.fromarray(cropped.astype(np.uint8))
    pil_resized = pil_crop.resize((28, 28), Image.LANCZOS)
    resized = np.array(pil_resized, dtype=np.float32)

    # 6. Normalise to match MNIST training distribution
    normalised = (resized / 255.0 - MNIST_MEAN) / MNIST_STD

    # 7. Return flat list
    return normalised.flatten().tolist()


def pixels_to_debug_png(pixels_784: list[float]) -> str:
    """
    Convert a list of 784 normalised floats back to a base64 PNG for debugging.
    Returns a data URI string.
    """
    arr = np.array(pixels_784, dtype=np.float32).reshape(28, 28)
    # un-normalise
    arr = (arr * MNIST_STD + MNIST_MEAN) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L').resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"