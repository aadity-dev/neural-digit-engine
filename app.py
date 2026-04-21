"""
Phase 3 — Flask REST API
=========================
Bridges the HTML/JS frontend with the C++ inference engine.

Install:
    pip install flask flask-cors

Run:
    python app.py

Endpoints:
    POST /predict   — receives 784 pixel values, returns prediction JSON
    GET  /health    — health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json

app = Flask(__name__)
CORS(app)  # allow frontend (different port) to call this API

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(BASE_DIR, '..', 'cpp', 'engine')

# ─────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    engine_exists = os.path.isfile(ENGINE_PATH)
    return jsonify({
        'status' : 'ok',
        'engine' : 'found' if engine_exists else 'NOT FOUND — run: g++ -O2 -o engine engine.cpp'
    })

# ─────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data or 'pixels' not in data:
            return jsonify({'error': 'Missing pixels field'}), 400

        pixels = data['pixels']

        if len(pixels) != 784:
            return jsonify({
                'error': f'Expected 784 pixel values, got {len(pixels)}'
            }), 400

        # Check engine binary exists
        if not os.path.isfile(ENGINE_PATH):
            return jsonify({
                'error': 'C++ engine not found. Build it first: g++ -O2 -o engine engine.cpp'
            }), 500

        # Build input string: 784 floats separated by spaces
        pixel_str = ' '.join(str(float(p)) for p in pixels)

        # Call C++ engine via subprocess
        result = subprocess.run(
            [ENGINE_PATH, '--stdin'],
            input=pixel_str,
            capture_output=True,
            text=True,
            timeout=10       # 10 second timeout
        )

        if result.returncode != 0:
            return jsonify({
                'error': 'Engine error: ' + result.stderr
            }), 500

        # Parse JSON output from C++ engine
        prediction = json.loads(result.stdout)

        return jsonify({
            'success'      : True,
            'digit'        : prediction['digit'],
            'confidence'   : round(prediction['confidence'], 2),
            'probabilities': prediction['probabilities']
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Engine timed out'}), 500
    except json.JSONDecodeError:
        return jsonify({'error': 'Engine returned invalid JSON: ' + result.stdout}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Phase 3 — Flask API starting")
    print("  POST http://localhost:5000/predict")
    print("  GET  http://localhost:5000/health")
    print("=" * 50)
    app.run(debug=True, port=5000)