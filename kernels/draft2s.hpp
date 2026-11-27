#pragma once

// Two-stage Act-PCA + Q4 preview + exact Q8 gather head (kernels/draft2s):
//   Stage 1: xr[T, R] = x[T, K] * E[R, K]^T            (BF16 -> BF16, low-rank orthonormal Act-PCA projection)
//   Stage 2: scores[T, V] = xr[T, R] * P_q4[V, R]^T    (BF16 x Q4 -> F32, compact or full preview)
//            2-pass 24-bit radix histogram -> up to `cap` exact candidate IDs per row (zero overflow)
//   Stage 3: exact Q8 dot products on those <= cap rows of W_q8[N_full, K], mapped via row_map (if non-null)
//            fused top-kLogitCands extraction + MtpPick (with p_top1 = 1 / sum_{cand} exp((v - max)/T))

#include "kernels/embedding.hpp"
#include "kernels/logits_topk.hpp"
#include "kernels/mtp_pick.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::kernels {

constexpr int32_t kDraft2SMaxCap = 2048;

struct Draft2SParams {
    const uint16_t *E_bf16 = nullptr;  // [R, K] BF16 on device
    Q4DeviceView P_q4{};               // [V, R] Q4 G64 preview on device
    Q8DeviceView W_q8{};               // [N_full, K] exact Q8 LM head on device
    const int32_t *row_map = nullptr;  // [V] local->orig vocab id on device, or nullptr for identity
    int64_t R = 0;                     // preview rank (multiple of P_q4.G, e.g. 512)
    int64_t V = 0;                     // preview vocab rows
    int32_t M_cand = 512;              // target candidate cutoff
    int32_t cap = 1024;                // hard candidate limit (M_cand <= cap <= kDraft2SMaxCap)
};

size_t draft2s_workspace_bytes(int64_t max_T, int64_t max_V, int64_t max_R, int32_t max_cap);

// x_bf16: [T, K] BF16 on device. n_valid: 1..p.V (rows beyond n_valid are masked out like logits_topk).
// inv_T_dev: [T] F32 on device (only required if out_picks != nullptr).
// err pointers may be nullptr if out_picks == nullptr; otherwise must point to valid device error slots.
// At least one of out_cands / out_picks must be non-null.
void draft2s_forward(const void *x_bf16, const Draft2SParams &p, int64_t T, int64_t n_valid, const float *inv_T_dev,
                     const uint32_t *router_err, const uint32_t *expert_err, const uint32_t *attn_err,
                     const EmbeddingError *emb_err, LogitCand *out_cands, uint32_t *out_cands_nan, MtpPick *out_picks,
                     void *ws, size_t ws_bytes, hipStream_t stream);

}  // namespace strix::kernels
