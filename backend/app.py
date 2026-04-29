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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import json

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.abspath(os.path.join(BASE_DIR, '..'))
ENGINE_PATH  = os.path.join(ROOT_DIR, 'engine')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')

app = Flask(__name__)
CORS(app)

# Serve frontend
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

# Health check
@app.route('/health', methods=['GET'])
def health():
    engine_exists = os.path.isfile(ENGINE_PATH)
    # Attempt to compile if missing
    if not engine_exists:
        cpp_source = os.path.join(ROOT_DIR, 'cpp', 'engine.cpp')
        if os.path.isfile(cpp_source):
            try:
                subprocess.run(['g++', '-O3', cpp_source, '-o', ENGINE_PATH], check=True)
                engine_exists = True
            except Exception as e:
                pass

    return jsonify({
        'status' : 'ok',
        'engine' : 'found' if engine_exists else 'missing',
        'frontend': os.path.isdir(FRONTEND_DIR)
    })

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'pixels' not in data:
            return jsonify({'error': 'Missing pixels field'}), 400

        pixels = data['pixels']
        if len(pixels) != 784:
            return jsonify({'error': f'Expected 784 pixels, got {len(pixels)}'}), 400

        if not os.path.isfile(ENGINE_PATH):
            cpp_source = os.path.join(ROOT_DIR, 'cpp', 'engine.cpp')
            if os.path.isfile(cpp_source):
                try:
                    subprocess.run(['g++', '-O3', cpp_source, '-o', ENGINE_PATH], check=True)
                except Exception as e:
                    return jsonify({'error': f'Failed to compile engine: {str(e)}'}), 500
            else:
                return jsonify({'error': f'Engine source not found at {cpp_source}'}), 500

        if not os.path.isfile(ENGINE_PATH):
            return jsonify({'error': f'Engine not found at {ENGINE_PATH}'}), 500

        pixel_str = ' '.join(str(float(p)) for p in pixels)
        result = subprocess.run(
            [ENGINE_PATH, '--stdin'],
            input=pixel_str,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return jsonify({'error': 'Engine error: ' + result.stderr}), 500

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
        return jsonify({'error': 'Engine returned invalid output'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"  ROOT     : {ROOT_DIR}")
    print(f"  FRONTEND : {FRONTEND_DIR}  exists={os.path.isdir(FRONTEND_DIR)}")
    print(f"  ENGINE   : {ENGINE_PATH}   exists={os.path.isfile(ENGINE_PATH)}")
    print("  Visit --> http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)