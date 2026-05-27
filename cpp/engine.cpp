/*
 * cpp/engine.cpp  — Phase 4 (CNN)
 * ================================
 * Architecture:
 *   Input  : 1 × 28 × 28  (784 floats from stdin)
 *   Conv1  : 32 filters, 3×3, padding=1  → 32 × 28 × 28  + ReLU
 *   Conv2  : 64 filters, 3×3, padding=1  → 64 × 28 × 28  + ReLU
 *   MaxPool: 2×2                          → 64 × 14 × 14
 *   Flatten: 12544
 *   FC1    : 12544 → 128                  + ReLU
 *   FC2    : 128   → 10                   + Softmax
 *
 * BN is pre-folded into Conv/FC weights by train.py — no BN needed here.
 *
 * Weight files (weights/):
 *   conv1_weight.txt   288     floats  (32 filters × 1 × 3 × 3)
 *   conv1_bias.txt     32      floats
 *   conv2_weight.txt   18432   floats  (64 × 32 × 3 × 3)
 *   conv2_bias.txt     64      floats
 *   fc1_weight.txt     1605632 floats  (128 × 12544)
 *   fc1_bias.txt       128     floats
 *   fc2_weight.txt     1280    floats  (10 × 128)
 *   fc2_bias.txt       10      floats
 *
 * Compile (macOS / Linux):
 *   g++ -O2 -std=c++17 -o engine engine.cpp
 *
 * Test:
 *   cat ../weights/test_image.txt | ./engine
 *   # Expected:  PREDICTION 7
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <numeric>

// ── Weight loading ────────────────────────────────────────────────────────────

static std::string find_weights_dir() {
    // Try common locations relative to CWD
    for (auto dir : {"weights", "../weights", "../../weights"}) {
        std::ifstream probe(std::string(dir) + "/conv1_bias.txt");
        if (probe.good()) return dir;
    }
    throw std::runtime_error(
        "Cannot find weights/ directory. Run train.py first.");
}

static std::vector<float> load_floats(const std::string& path, int expected = -1) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open weight file: " + path);
    std::vector<float> v;
    float x;
    while (f >> x) v.push_back(x);
    if (expected >= 0 && (int)v.size() != expected)
        throw std::runtime_error(
            "Size mismatch in " + path +
            ": expected " + std::to_string(expected) +
            ", got " + std::to_string(v.size()) +
            ". Re-run train.py.");
    return v;
}

// ── Activation ───────────────────────────────────────────────────────────────

static inline float relu(float x) { return x > 0.f ? x : 0.f; }

static std::vector<float> softmax(const std::vector<float>& z) {
    float mx = *std::max_element(z.begin(), z.end());
    std::vector<float> out(z.size());
    float sum = 0;
    for (size_t i = 0; i < z.size(); ++i) {
        out[i] = std::exp(z[i] - mx);
        sum   += out[i];
    }
    for (auto& v : out) v /= sum;
    return out;
}

// ── Conv2d forward  (single-threaded, no im2col — simple but correct) ────────
//
// input  : (in_ch, H, W)  — flat, row-major: [ch][row][col]
// weight : (out_ch × in_ch × kH × kW)  — same order as PyTorch
// bias   : (out_ch,)
// padding: same as PyTorch padding=1 for 3×3 → output same H×W
// output : (out_ch, H_out, W_out)

static std::vector<float> conv2d(
    const std::vector<float>& input,
    int in_ch, int H, int W,
    const std::vector<float>& weight,   // out_ch × in_ch × kH × kW
    const std::vector<float>& bias,
    int out_ch, int kH, int kW,
    int pad,
    bool apply_relu)
{
    int H_out = H + 2*pad - kH + 1;
    int W_out = W + 2*pad - kW + 1;
    std::vector<float> output(out_ch * H_out * W_out, 0.f);

    // weight index: [oc][ic][ky][kx] = oc*(in_ch*kH*kW) + ic*(kH*kW) + ky*kW + kx
    int ksize   = kH * kW;
    int wstride = in_ch * ksize;

    for (int oc = 0; oc < out_ch; ++oc) {
        float b = bias[oc];
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                float acc = b;
                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int ky = 0; ky < kH; ++ky) {
                        int ih = oh + ky - pad;
                        if (ih < 0 || ih >= H) continue;
                        for (int kx = 0; kx < kW; ++kx) {
                            int iw = ow + kx - pad;
                            if (iw < 0 || iw >= W) continue;
                            float in_val = input[ic * H * W + ih * W + iw];
                            float w_val  = weight[oc * wstride + ic * ksize + ky * kW + kx];
                            acc += in_val * w_val;
                        }
                    }
                }
                float val = apply_relu ? relu(acc) : acc;
                output[oc * H_out * W_out + oh * W_out + ow] = val;
            }
        }
    }
    return output;
}

// ── MaxPool2d (2×2, stride=2) ─────────────────────────────────────────────────

static std::vector<float> maxpool2d(
    const std::vector<float>& input,
    int ch, int H, int W)
{
    int H_out = H / 2, W_out = W / 2;
    std::vector<float> output(ch * H_out * W_out);
    for (int c = 0; c < ch; ++c) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                int ih = oh * 2, iw = ow * 2;
                float mx = input[c*H*W + ih*W + iw];
                mx = std::max(mx, input[c*H*W + ih*W     + iw+1]);
                mx = std::max(mx, input[c*H*W + (ih+1)*W + iw  ]);
                mx = std::max(mx, input[c*H*W + (ih+1)*W + iw+1]);
                output[c * H_out * W_out + oh * W_out + ow] = mx;
            }
        }
    }
    return output;
}

// ── Fully-connected (dense) ───────────────────────────────────────────────────

static std::vector<float> dense(
    const std::vector<float>& x,
    const std::vector<float>& W,   // (out × in) flat
    const std::vector<float>& b,
    int out_size,
    bool apply_relu)
{
    int in_size = (int)x.size();
    std::vector<float> y(out_size);
    for (int o = 0; o < out_size; ++o) {
        float acc = b[o];
        const float* row = &W[o * in_size];
        for (int i = 0; i < in_size; ++i)
            acc += row[i] * x[i];
        y[o] = apply_relu ? relu(acc) : acc;
    }
    return y;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    try {
        std::string wdir = find_weights_dir();

        // Load all weights
        // conv1: 32 filters, 1 in_ch, 3×3 → 32*1*9 = 288
        auto conv1_W = load_floats(wdir + "/conv1_weight.txt", 32 * 1 * 9);
        auto conv1_b = load_floats(wdir + "/conv1_bias.txt",   32);
        // conv2: 64 filters, 32 in_ch, 3×3 → 64*32*9 = 18432
        auto conv2_W = load_floats(wdir + "/conv2_weight.txt", 64 * 32 * 9);
        auto conv2_b = load_floats(wdir + "/conv2_bias.txt",   64);
        // fc1: 128 × 12544
        auto fc1_W   = load_floats(wdir + "/fc1_weight.txt",   128 * 3136);
        auto fc1_b   = load_floats(wdir + "/fc1_bias.txt",     128);
        // fc2: 10 × 128
        auto fc2_W   = load_floats(wdir + "/fc2_weight.txt",   10 * 128);
        auto fc2_b   = load_floats(wdir + "/fc2_bias.txt",     10);

        // Read 784 normalised floats from stdin
        std::vector<float> input;
        input.reserve(784);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try { input.push_back(std::stof(line)); }
            catch (...) { throw std::runtime_error("Bad input value: '" + line + "'"); }
        }
        if ((int)input.size() != 784)
            throw std::runtime_error(
                "Expected 784 pixels, got " + std::to_string(input.size()));

        // ── Forward pass ──────────────────────────────────────────────────

        // input shape: (1, 28, 28)
        auto h1 = conv2d(input,
                         /*in_ch=*/1, /*H=*/28, /*W=*/28,
                         conv1_W, conv1_b,
                         /*out_ch=*/32, /*kH=*/3, /*kW=*/3,
                         /*pad=*/1, /*relu=*/true);
        // h1 shape: (32, 28, 28)

        auto h2 = conv2d(h1,
                         /*in_ch=*/32, /*H=*/28, /*W=*/28,
                         conv2_W, conv2_b,
                         /*out_ch=*/64, /*kH=*/3, /*kW=*/3,
                         /*pad=*/1, /*relu=*/true);
        // h2 shape: (64, 28, 28)

        auto h3 = maxpool2d(h2, /*ch=*/64, /*H=*/28, /*W=*/28);
        // h3 shape: (64, 14, 14) = 12544 values

        auto h4     = dense(h3,  fc1_W, fc1_b, 128, /*relu=*/true);
        auto logits = dense(h4,  fc2_W, fc2_b, 10,  /*relu=*/false);
        auto probs  = softmax(logits);

        int prediction = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

        std::cout << "PREDICTION " << prediction << "\n";
        std::cout << "SCORES";
        for (float p : probs) std::cout << " " << p;
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ENGINE ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}