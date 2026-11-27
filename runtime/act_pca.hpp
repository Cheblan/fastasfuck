#pragma once

// Zero-dependency FP64 dual-Gram eigensolver for Activation PCA basis construction + disk binary cache.
// Given sampled hidden states U [N, K] (BF16), returns orthonormal rows E [R, K] (BF16) spanning the top R
// right singular vectors of U, used by Two-Stage draft/LM heads (kernels/draft2s).

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

// Computes top-R orthonormal principal components E [R, K] (BF16) from u_bf16 [N, K].
// Stride-samples to at most 1024 rows before forming the N x N dual Gram matrix. Thread-parallel across CPU cores.
std::vector<uint16_t> compute_act_pca_basis_bf16(const uint16_t *u_bf16, int64_t N, int64_t K, int64_t R);

// Atomic cache persistence keyed by weights fingerprint, rank R, and hidden dimension K.
bool load_act_pca_cache(const std::string &path, const std::string &fingerprint, int64_t R, int64_t K,
                        std::vector<uint16_t> &out_bf16);
void save_act_pca_cache(const std::string &path, const std::string &fingerprint, int64_t R, int64_t K,
                        const std::vector<uint16_t> &e_bf16);

// Rich multi-domain calibration text (~1024 tokens: chat template, Python async, LaTeX math, Linux internals,
// Russian technical prose, tool call JSON, and <think> reasoning trace).
std::string default_act_pca_calibration_text();

}  // namespace strix
