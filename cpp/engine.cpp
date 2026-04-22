/*
 * Phase 2 — C++ Inference Engine
 * ================================
 * Reads trained weights from .txt files and predicts
 * handwritten digits (0-9) from 784 pixel values.
 *
 * Architecture:  784 → ReLU(128) → Softmax(10)
 *
 * Build:
 *   g++ -O2 -o engine engine.cpp
 *
 * Usage (predict from test_image.txt):
 *   ./engine
 *
 * Usage (predict from stdin — used by Flask):
 *   echo "0.1 0.5 ..." | ./engine --stdin
 *
 * Usage (verify against test_label.txt):
 *   ./engine --verify
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iomanip>

// ─────────────────────────────────────────
//  CONFIG — paths relative to this binary
// ─────────────────────────────────────────
const std::string WEIGHTS_DIR  = "./weights/";
const std::string FC1_W        = WEIGHTS_DIR + "fc1_weight.txt";
const std::string FC1_B        = WEIGHTS_DIR + "fc1_bias.txt";
const std::string FC2_W        = WEIGHTS_DIR + "fc2_weight.txt";
const std::string FC2_B        = WEIGHTS_DIR + "fc2_bias.txt";
const std::string TEST_IMAGE   = WEIGHTS_DIR + "test_image.txt";
const std::string TEST_LABEL   = WEIGHTS_DIR + "test_label.txt";

// ─────────────────────────────────────────
//  LOAD HELPERS
// ─────────────────────────────────────────

// Load a flat vector of floats (one per line)
std::vector<float> load_vector(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<float> v;
    float x;
    while (f >> x) v.push_back(x);
    return v;
}

// Load a 2-D matrix stored as rows of space-separated floats
// Returns flat row-major vector; sets rows and cols
std::vector<float> load_matrix(const std::string& path,
                                int& rows, int& cols) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);

    std::vector<float> data;
    std::string line;
    rows = 0;
    cols = -1;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<float> row;
        float x;
        while (ss >> x) row.push_back(x);
        if (!row.empty()) {
            if (cols == -1) cols = (int)row.size();
            data.insert(data.end(), row.begin(), row.end());
            rows++;
        }
    }
    return data;
}

// ─────────────────────────────────────────
//  MATH PRIMITIVES
// ─────────────────────────────────────────

// output[i] = bias[i] + sum_j(weight[i*in_size + j] * input[j])
std::vector<float> linear(const std::vector<float>& input,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_size, int in_size) {
    std::vector<float> out(out_size);
    for (int i = 0; i < out_size; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_size; j++)
            sum += weight[i * in_size + j] * input[j];
        out[i] = sum;
    }
    return out;
}

// ReLU: max(0, x)
void relu(std::vector<float>& v) {
    for (auto& x : v)
        if (x < 0.0f) x = 0.0f;
}

// Softmax: converts logits to probabilities
std::vector<float> softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_v(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        exp_v[i] = std::exp(logits[i] - max_val);  // stable softmax
        sum += exp_v[i];
    }
    for (auto& x : exp_v) x /= sum;
    return exp_v;
}

// ─────────────────────────────────────────
//  INFERENCE ENGINE
// ─────────────────────────────────────────
struct Prediction {
    int   digit;
    float confidence;
    std::vector<float> all_probs;   // probabilities for digits 0-9
};

Prediction predict(const std::vector<float>& pixels) {
    // Load weights once (in production you'd cache these)
    int r1, c1, r2, c2;
    std::vector<float> fc1_w = load_matrix(FC1_W, r1, c1); // 128x784
    std::vector<float> fc1_b = load_vector(FC1_B);          // 128
    std::vector<float> fc2_w = load_matrix(FC2_W, r2, c2); // 10x128
    std::vector<float> fc2_b = load_vector(FC2_B);          // 10

    // Forward pass
    // Layer 1: linear → ReLU
    std::vector<float> h = linear(pixels, fc1_w, fc1_b, r1, c1);
    relu(h);

    // Layer 2: linear → Softmax
    std::vector<float> logits = linear(h, fc2_w, fc2_b, r2, c2);
    std::vector<float> probs  = softmax(logits);

    // Pick highest probability
    int best = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

    return { best, probs[best] * 100.0f, probs };
}

// ─────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────
int main(int argc, char* argv[]) {

    std::string mode = "file";   // default: read test_image.txt
    if (argc > 1) mode = argv[1];

    try {
        std::vector<float> pixels;

        // ── Mode 1: read from test_image.txt ──────────────
        if (mode == "file" || mode == "--verify") {
            pixels = load_vector(TEST_IMAGE);
            if (pixels.size() != 784)
                throw std::runtime_error("test_image.txt must have 784 values");
        }

        // ── Mode 2: read 784 floats from stdin ─────────────
        else if (mode == "--stdin") {
            float x;
            while (std::cin >> x) pixels.push_back(x);
            if (pixels.size() != 784)
                throw std::runtime_error("stdin must supply exactly 784 values");
        }

        else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

        // ── Run inference ───────────────────────────────────
        Prediction p = predict(pixels);

        // ── Output ──────────────────────────────────────────
        std::cout << "{\n";
        std::cout << "  \"digit\": "      << p.digit                              << ",\n";
        std::cout << "  \"confidence\": " << std::fixed << std::setprecision(2)
                                          << p.confidence                         << ",\n";
        std::cout << "  \"probabilities\": [";
        for (int i = 0; i < 10; i++) {
            std::cout << std::fixed << std::setprecision(4) << p.all_probs[i];
            if (i < 9) std::cout << ", ";
        }
        std::cout << "]\n}\n";

        // ── Verify mode ─────────────────────────────────────
        if (mode == "--verify") {
            std::ifstream lf(TEST_LABEL);
            int expected = -1;
            if (lf.is_open()) lf >> expected;

            std::cerr << "\n=== VERIFICATION ===\n";
            std::cerr << "Expected : " << expected << "\n";
            std::cerr << "Got      : " << p.digit  << "\n";
            std::cerr << (p.digit == expected ? "PASS ✓" : "FAIL ✗") << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}