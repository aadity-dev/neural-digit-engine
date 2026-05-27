/*
 * cpp/engine.cpp  — Phase 5 (Binary weights)
 * ============================================
 * Same CNN architecture as Phase 4:
 *   Input → Conv(32)+Pool → Conv(64)+Pool → FC(128) → 10 → Softmax
 *
 * Phase 5 change: loads .bin files (raw float32) instead of .txt
 *   - Single fread() call per weight file
 *   - Load time: ~5ms vs ~200ms for .txt
 *   - Falls back to .txt automatically if .bin not found
 *     (so it still works before running export_binary.py)
 *
 * Compile:
 *   g++ -O2 -std=c++17 -o engine engine.cpp
 *
 * Test:
 *   cat ../weights/test_image.txt | ./engine
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <chrono>

// ── Weight loading ────────────────────────────────────────────────────────────

static std::string find_weights_dir() {
    for (auto dir : {"weights", "../weights", "../../weights"}) {
        // check for either .bin or .txt
        std::ifstream probe(std::string(dir) + "/conv1_bias.bin");
        if (probe.good()) return dir;
        std::ifstream probe2(std::string(dir) + "/conv1_bias.txt");
        if (probe2.good()) return dir;
    }
    throw std::runtime_error("Cannot find weights/. Run exportWeights.py then export_binary.py.");
}

// Load from binary (.bin) — fast path
static std::vector<float> load_bin(const std::string& path, int expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};   // signal: file not found, try .txt

    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    int count = (int)(bytes / sizeof(float));
    if (count != expected)
        throw std::runtime_error(
            path + ": expected " + std::to_string(expected) +
            " floats (" + std::to_string(expected*4) + " bytes), got " +
            std::to_string(count) + " floats (" + std::to_string(bytes) + " bytes).\n"
            "Re-run export_binary.py.");

    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// Load from text (.txt) — fallback
static std::vector<float> load_txt(const std::string& path, int expected) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path +
            "\nRun exportWeights.py first.");
    std::vector<float> v;
    v.reserve(expected);
    float x;
    while (f >> x) v.push_back(x);
    if ((int)v.size() != expected)
        throw std::runtime_error(
            path + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(v.size()));
    return v;
}

// Try .bin first, fall back to .txt
static std::vector<float> load_weights(
    const std::string& wdir, const std::string& name, int expected)
{
    std::string bin_path = wdir + "/" + name + ".bin";
    auto v = load_bin(bin_path, expected);
    if (!v.empty()) return v;

    // .bin not found — try .txt (pre-Phase-5 fallback)
    std::string txt_path = wdir + "/" + name + ".txt";
    std::cerr << "  [warn] " << name << ".bin not found, using .txt (run export_binary.py for speed)\n";
    return load_txt(txt_path, expected);
}

// ── Activations ───────────────────────────────────────────────────────────────

static inline float relu(float x) { return x > 0.f ? x : 0.f; }

static std::vector<float> softmax(const std::vector<float>& z) {
    float mx = *std::max_element(z.begin(), z.end());
    std::vector<float> out(z.size());
    float sum = 0;
    for (size_t i = 0; i < z.size(); ++i) { out[i] = std::exp(z[i] - mx); sum += out[i]; }
    for (auto& v : out) v /= sum;
    return out;
}

// ── Conv2d (3×3, padding=1, stride=1) ────────────────────────────────────────

static std::vector<float> conv2d(
    const std::vector<float>& input,
    int in_ch, int H, int W,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int out_ch)
{
    const int kH = 3, kW = 3, pad = 1;
    std::vector<float> output(out_ch * H * W);
    int ksize = kH * kW, wstride = in_ch * ksize;

    for (int oc = 0; oc < out_ch; ++oc) {
        float b = bias[oc];
        for (int oh = 0; oh < H; ++oh) {
            for (int ow = 0; ow < W; ++ow) {
                float acc = b;
                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int ky = 0; ky < kH; ++ky) {
                        int ih = oh + ky - pad;
                        if (ih < 0 || ih >= H) continue;
                        for (int kx = 0; kx < kW; ++kx) {
                            int iw = ow + kx - pad;
                            if (iw < 0 || iw >= W) continue;
                            acc += input[ic*H*W + ih*W + iw]
                                 * weight[oc*wstride + ic*ksize + ky*kW + kx];
                        }
                    }
                }
                output[oc*H*W + oh*W + ow] = relu(acc);
            }
        }
    }
    return output;
}

// ── MaxPool2d (2×2, stride=2) ─────────────────────────────────────────────────

static std::vector<float> maxpool2d(
    const std::vector<float>& input, int ch, int H, int W)
{
    int Ho = H/2, Wo = W/2;
    std::vector<float> out(ch * Ho * Wo);
    for (int c = 0; c < ch; ++c)
        for (int oh = 0; oh < Ho; ++oh)
            for (int ow = 0; ow < Wo; ++ow) {
                int ih = oh*2, iw = ow*2;
                float mx = input[c*H*W + ih*W + iw];
                mx = std::max(mx, input[c*H*W + ih*W     + iw+1]);
                mx = std::max(mx, input[c*H*W + (ih+1)*W + iw  ]);
                mx = std::max(mx, input[c*H*W + (ih+1)*W + iw+1]);
                out[c*Ho*Wo + oh*Wo + ow] = mx;
            }
    return out;
}

// ── Dense ─────────────────────────────────────────────────────────────────────

static std::vector<float> dense(
    const std::vector<float>& x,
    const std::vector<float>& W,
    const std::vector<float>& b,
    int out_size, bool apply_relu)
{
    int in_size = (int)x.size();
    std::vector<float> y(out_size);
    for (int o = 0; o < out_size; ++o) {
        float acc = b[o];
        const float* row = &W[o * in_size];
        for (int i = 0; i < in_size; ++i) acc += row[i] * x[i];
        y[o] = apply_relu ? relu(acc) : acc;
    }
    return y;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::string wdir = find_weights_dir();

        // Load weights — tries .bin first, falls back to .txt
        auto conv1_W = load_weights(wdir, "conv1_weight", 32 * 1 * 9);
        auto conv1_b = load_weights(wdir, "conv1_bias",   32);
        auto conv2_W = load_weights(wdir, "conv2_weight", 64 * 32 * 9);
        auto conv2_b = load_weights(wdir, "conv2_bias",   64);
        auto fc1_W   = load_weights(wdir, "fc1_weight",   128 * 3136);
        auto fc1_b   = load_weights(wdir, "fc1_bias",     128);
        auto fc2_W   = load_weights(wdir, "fc2_weight",   10 * 128);
        auto fc2_b   = load_weights(wdir, "fc2_bias",     10);

        auto t1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Read 784 floats from stdin
        std::vector<float> input;
        input.reserve(784);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try { input.push_back(std::stof(line)); }
            catch (...) { throw std::runtime_error("Bad value: '" + line + "'"); }
        }
        if ((int)input.size() != 784)
            throw std::runtime_error(
                "Expected 784 pixels, got " + std::to_string(input.size()));

        // ── Forward pass ──────────────────────────────────────────────────
        auto h1 = conv2d(input, 1,  28, 28, conv1_W, conv1_b, 32);
        auto h2 = maxpool2d(h1, 32, 28, 28);
        auto h3 = conv2d(h2,   32, 14, 14, conv2_W, conv2_b, 64);
        auto h4 = maxpool2d(h3, 64, 14, 14);
        auto h5     = dense(h4, fc1_W, fc1_b, 128, true);
        auto logits = dense(h5, fc2_W, fc2_b, 10,  false);
        auto probs  = softmax(logits);

        auto t2 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        int pred = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());

        std::cout << "PREDICTION " << pred << "\n";
        std::cout << "SCORES";
        for (float p : probs) std::cout << " " << p;
        std::cout << "\n";
        std::cerr << "load=" << load_ms << "ms  infer=" << infer_ms << "ms\n";

    } catch (const std::exception& e) {
        std::cerr << "ENGINE ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}