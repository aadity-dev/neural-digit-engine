"""
backend/app.py  — Phase 2
==========================
Changes from Phase 1:
  - Uses preprocess.canvas_to_mnist() for correct MNIST-style input
  - Calls C++ engine with 784 normalised floats
  - Returns top-3 predictions + all 10 softmax scores for confidence bars
  - /debug endpoint returns what the engine actually "sees" (28×28 PNG)
  - CORS header added so frontend can call from file:// during dev
"""

import subprocess
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocess import canvas_to_mnist, pixels_to_debug_png

app = Flask(__name__)
CORS(app)   # allow canvas POST from any origin (safe — no auth involved)

# Path to the compiled C++ engine binary
ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'cpp', 'engine'
)
WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'weights'
)


def run_engine(pixels: list[float]) -> dict:
    """
    Write pixels to a temp file, run the C++ engine, parse its output.

    The C++ engine reads 784 floats from stdin (one per line) and prints:
        PREDICTION <digit>
        SCORES <s0> <s1> ... <s9>

    Returns:
        { "prediction": int, "scores": [float x10], "top3": [(digit, score), ...] }
    """
    pixel_input = '\n'.join(f"{v:.8f}" for v in pixels)

    try:
        result = subprocess.run(
            [ENGINE_PATH],
            input=pixel_input,
            capture_output=True,
            text=True,
            timeout=5
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"C++ engine not found at {ENGINE_PATH}. "
            "Run: cd cpp && g++ -O2 -o engine engine.cpp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("C++ engine timed out after 5 seconds")

    if result.returncode != 0:
        raise RuntimeError(f"Engine error: {result.stderr.strip()}")

    # Parse engine output
    prediction = None
    scores = None
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith('PREDICTION'):
            prediction = int(line.split()[1])
        elif line.startswith('SCORES'):
            scores = [float(x) for x in line.split()[1:]]

    if prediction is None or scores is None:
        raise RuntimeError(f"Unexpected engine output:\n{result.stdout}")

    # Build top-3
    top3 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    top3 = [(int(d), round(float(s) * 100, 1)) for d, s in top3]

    return {
        "prediction": prediction,
        "scores": [round(float(s) * 100, 2) for s in scores],
        "top3": top3
    }


@app.route('/')
def index():
    return jsonify({"status": "Neural Digit Engine — Phase 2", "engine": ENGINE_PATH})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body: { "image": "<base64 PNG data URI>" }
    Returns:  {
        "prediction": 7,
        "confidence": 98.4,
        "scores": [0.1, 0.2, ...],   // 10 values, sum ≈ 100
        "top3": [[7, 98.4], [1, 1.2], [9, 0.4]],
        "debug_image": "data:image/png;base64,..."  // what the engine saw
    }
    """
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' field in request body"}), 400

    try:
        pixels = canvas_to_mnist(data['image'])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = run_engine(pixels)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # add debug image showing what the engine saw
    result["debug_image"] = pixels_to_debug_png(pixels)
    result["confidence"]  = result["scores"][result["prediction"]]

    return jsonify(result)


@app.route('/debug', methods=['POST'])
def debug():
    """
    Same as /predict but only returns the preprocessed 28×28 image.
    Useful for checking the preprocessing pipeline in isolation.
    """
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' field"}), 400
    try:
        pixels = canvas_to_mnist(data['image'])
        return jsonify({"debug_image": pixels_to_debug_png(pixels)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"  Neural Digit Engine — Phase 2  (port {port})")
    print(f"  Engine: {ENGINE_PATH}")
    print(f"  Weights: {WEIGHTS_DIR}")
    app.run(host='0.0.0.0', port=port, debug=False)