#include "kernels/draft2s.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/types.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kBins = 4096;
constexpr int kThreads = 256;
constexpr int kGatherThreads = 256;

constexpr size_t align_up(size_t n, size_t align = 256) { return (n + align - 1) & ~(align - 1); }

__device__ __forceinline__ uint32_t order_key(float v) {
    uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int32_t find_suffix_thr(const uint32_t *ghist, int32_t target, uint32_t *wave_sums,
                                                   int32_t *s_best, uint32_t *s_above) {
    if (threadIdx.x == 0) {
        *s_best = 0;
        *s_above = 0;
    }
    constexpr int kPer = kBins / kThreads;  // 16
    const int j = threadIdx.x;
    const int base = (kThreads - 1 - j) * kPer;
    uint32_t own = 0;
    for (int b = 0; b < kPer; ++b) own += ghist[base + b];

    const int lane = j % warpSize;
    const int wave = j / warpSize;
    uint32_t inc = own;
    for (int off = 1; off < warpSize; off <<= 1) {
        uint32_t o = (uint32_t)__shfl_up((int)inc, off);
        if (lane >= off) inc += o;
    }
    if (lane == warpSize - 1) wave_sums[wave] = inc;
    __syncthreads();
    uint32_t before = 0;
    for (int w = 0; w < wave; ++w) before += wave_sums[w];
    uint32_t suf_before = before + inc - own;
    if (suf_before < (uint32_t)target && suf_before + own >= (uint32_t)target) {
        uint32_t acc = suf_before;
        for (int b = kPer - 1; b >= 0; --b) {
            const uint32_t c = ghist[base + b];
            if (acc + c >= (uint32_t)target) {
                *s_best = base + b;
                *s_above = acc;
                break;
            }
            acc += c;
        }
    }
    __syncthreads();
    return *s_best;
}

__global__ void __launch_bounds__(kThreads)
    preview_zero_kernel(uint32_t *__restrict__ hist1, uint32_t *__restrict__ hist2, int32_t *__restrict__ cnt_gt,
                        int32_t *__restrict__ cnt_eq, uint32_t *__restrict__ row_nan, int64_t T) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_hist = T * kBins;
    for (int64_t i = tid; i < total_hist; i += (int64_t)gridDim.x * blockDim.x) {
        hist1[i] = 0u;
        hist2[i] = 0u;
    }
    if (tid < T) {
        cnt_gt[tid] = 0;
        cnt_eq[tid] = 0;
        row_nan[tid] = 0u;
    }
}

__global__ void __launch_bounds__(kThreads)
    preview_hist1_kernel(const float *__restrict__ scores, int64_t ld, int64_t n_valid, uint32_t *__restrict__ hist1,
                         uint32_t *__restrict__ row_nan) {
    __shared__ uint32_t lhist[kBins];
    __shared__ uint32_t lnan;
    for (int i = threadIdx.x; i < kBins; i += kThreads) lhist[i] = 0u;
    if (threadIdx.x == 0) lnan = 0u;
    __syncthreads();

    const int64_t t = blockIdx.y;
    const float *row = scores + t * ld;
    const int64_t stride = (int64_t)gridDim.x * kThreads;
    uint32_t my_nan = 0u;
    for (int64_t i = (int64_t)blockIdx.x * kThreads + threadIdx.x; i < n_valid; i += stride) {
        const float v = row[i];
        if (isnan(v)) {
            my_nan = 1u;
        } else {
            atomicAdd(&lhist[order_key(v) >> 20], 1u);
        }
    }
    if (my_nan) atomicOr(&lnan, 1u);
    __syncthreads();
    uint32_t *ghist = hist1 + t * kBins;
    for (int i = threadIdx.x; i < kBins; i += kThreads) {
        const uint32_t c = lhist[i];
        if (c) atomicAdd(&ghist[i], c);
    }
    if (threadIdx.x == 0 && lnan) atomicOr(&row_nan[t], 1u);
}

__global__ void __launch_bounds__(kThreads)
    preview_thresh1_kernel(const uint32_t *__restrict__ hist1, int32_t M_cand, int32_t *__restrict__ thr1_out,
                           int32_t *__restrict__ need2_out, int32_t *__restrict__ gt1_out) {
    __shared__ uint32_t wave_sums[kThreads / 32];
    __shared__ int32_t s_best;
    __shared__ uint32_t s_above;
    const int t = blockIdx.x;
    const int32_t b1 = find_suffix_thr(hist1 + (int64_t)t * kBins, M_cand, wave_sums, &s_best, &s_above);
    if (threadIdx.x == 0) {
        thr1_out[t] = b1;
        gt1_out[t] = (int32_t)s_above;
        need2_out[t] = max(1, M_cand - (int32_t)s_above);
    }
}

__global__ void __launch_bounds__(kThreads)
    preview_hist2_kernel(const float *__restrict__ scores, int64_t ld, int64_t n_valid,
                         const int32_t *__restrict__ thr1_in, uint32_t *__restrict__ hist2) {
    __shared__ uint32_t lhist[kBins];
    for (int i = threadIdx.x; i < kBins; i += kThreads) lhist[i] = 0u;
    __syncthreads();

    const int64_t t = blockIdx.y;
    const uint32_t thr1 = (uint32_t)thr1_in[t];
    const float *row = scores + t * ld;
    const int64_t stride = (int64_t)gridDim.x * kThreads;
    for (int64_t i = (int64_t)blockIdx.x * kThreads + threadIdx.x; i < n_valid; i += stride) {
        const float v = row[i];
        if (!isnan(v)) {
            const uint32_t k = order_key(v);
            if ((k >> 20) == thr1) {
                atomicAdd(&lhist[(k >> 8) & 0xFFFu], 1u);
            }
        }
    }
    __syncthreads();
    uint32_t *ghist = hist2 + t * kBins;
    for (int i = threadIdx.x; i < kBins; i += kThreads) {
        const uint32_t c = lhist[i];
        if (c) atomicAdd(&ghist[i], c);
    }
}

__global__ void __launch_bounds__(kThreads)
    preview_thresh2_kernel(const uint32_t *__restrict__ hist2, const int32_t *__restrict__ thr1_in,
                           const int32_t *__restrict__ need2_in, const int32_t *__restrict__ gt1_in, int32_t cap,
                           int32_t *__restrict__ thr24_out, int32_t *__restrict__ gt_total_out,
                           int32_t *__restrict__ cand_cnt_out) {
    __shared__ uint32_t wave_sums[kThreads / 32];
    __shared__ int32_t s_best;
    __shared__ uint32_t s_above;
    const int t = blockIdx.x;
    const uint32_t *ghist = hist2 + (int64_t)t * kBins;
    const int32_t b2 = find_suffix_thr(ghist, need2_in[t], wave_sums, &s_best, &s_above);
    if (threadIdx.x == 0) {
        const int32_t gt_tot = gt1_in[t] + (int32_t)s_above;
        const int32_t eq_tot = (int32_t)ghist[b2];
        thr24_out[t] = (thr1_in[t] << 12) | b2;
        gt_total_out[t] = gt_tot;
        cand_cnt_out[t] = min(cap, gt_tot + eq_tot);
    }
}

__global__ void __launch_bounds__(kThreads)
    preview_compact_kernel(const float *__restrict__ scores, int64_t ld, int64_t n_valid,
                           const int32_t *__restrict__ thr24_in, const int32_t *__restrict__ gt_total_in, int32_t cap,
                           int32_t *__restrict__ cand_idx, int32_t *__restrict__ cnt_gt, int32_t *__restrict__ cnt_eq) {
    const int64_t t = blockIdx.y;
    const uint32_t thr24 = (uint32_t)thr24_in[t];
    const int32_t gt_tot = gt_total_in[t];
    const float *row = scores + t * ld;
    int32_t *out = cand_idx + t * cap;
    int32_t *p_gt = cnt_gt + t;
    int32_t *p_eq = cnt_eq + t;
    const int64_t stride = (int64_t)gridDim.x * kThreads;
    for (int64_t i = (int64_t)blockIdx.x * kThreads + threadIdx.x; i < n_valid; i += stride) {
        const float v = row[i];
        if (!isnan(v)) {
            const uint32_t k24 = order_key(v) >> 8;
            if (k24 > thr24) {
                const int32_t pos = atomicAdd(p_gt, 1);
                if (pos < cap) out[pos] = (int32_t)i;
            } else if (k24 == thr24) {
                const int32_t pos = gt_tot + atomicAdd(p_eq, 1);
                if (pos < cap) out[pos] = (int32_t)i;
            }
        }
    }
}

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
__device__ __forceinline__ uint32_t q8_code_pair(const uint32_t (&words)[4], int p) {
    const uint32_t w = words[p / 2], sh = 16 * (p % 2);
    const float lo = (float)((w >> sh) & 0xFFu), hi = (float)((w >> (sh + 8)) & 0xFFu);
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}

__global__ void __launch_bounds__(kGatherThreads)
    q8_gather_rows_kernel(const uint16_t *__restrict__ x, const Q8DeviceView w, const int32_t *__restrict__ row_map,
                          const int32_t *__restrict__ cand_idx, const int32_t *__restrict__ cand_cnt, int32_t cap,
                          const uint32_t *__restrict__ row_nan_in, uint32_t *__restrict__ row_nan_out,
                          float *__restrict__ out_vals, int32_t *__restrict__ out_orig_ids) {
    __shared__ float wave_sum[kGatherThreads / 32];
    const int m = blockIdx.x;
    const int t = blockIdx.y;
    if (row_nan_in[t]) return;
    const int n_live = min(cand_cnt[t], cap);
    if (m >= n_live) return;

    const int32_t local_id = cand_idx[(int64_t)t * cap + m];
    const int32_t orig_id = row_map ? row_map[local_id] : local_id;

    float partial = 0.0f;
    const int64_t n_chunks = w.K / 16;
    const int64_t n_grp = w.K / w.G;
    constexpr uint32_t kOnePair = 0x3F803F80u;
    for (int64_t lane_chunk = threadIdx.x; lane_chunk < n_chunks; lane_chunk += blockDim.x) {
        const int64_t k = lane_chunk * 16;
        const uint4 *xp = reinterpret_cast<const uint4 *>(x + (int64_t)t * w.K + k);
        const uint4 u0 = xp[0];
        const uint4 u1 = xp[1];
        const uint4 packed = reinterpret_cast<const uint4 *>(w.q + (int64_t)orig_id * w.K)[lane_chunk];
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        const int64_t grp = k / w.G;
        const float s = __uint_as_float((uint32_t)w.scale[(int64_t)orig_id * n_grp + grp] << 16);
        const float mn = __uint_as_float((uint32_t)w.minv[(int64_t)orig_id * n_grp + grp] << 16);

        const uint32_t xw[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
        float xq = 0.f, sx = 0.f;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            xq = dot2(xw[p], q8_code_pair(words, p), xq);
            sx = dot2(xw[p], kOnePair, sx);
        }
        partial += s * xq + mn * sx;
    }

    for (int off = warpSize / 2; off > 0; off >>= 1) {
        partial += __shfl_xor(partial, off);
    }
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    if (lane == 0) wave_sum[wave] = partial;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x / warpSize;
        float tot = 0.f;
        for (int wi = 0; wi < nw; ++wi) tot += wave_sum[wi];
        if (isnan(tot)) {
            atomicOr(&row_nan_out[t], 1u);
        }
        out_vals[(int64_t)t * cap + m] = tot;
        out_orig_ids[(int64_t)t * cap + m] = orig_id;
    }
}

__global__ void __launch_bounds__(kThreads)
    gather_top20_and_pick_kernel(const float *__restrict__ vals, const int32_t *__restrict__ orig_ids,
                                 const int32_t *__restrict__ cand_cnt, const uint32_t *__restrict__ row_nan,
                                 int32_t cap, const float *__restrict__ inv_T_dev, const uint32_t *__restrict__ router_err,
                                 const uint32_t *__restrict__ expert_err, const uint32_t *__restrict__ attn_err,
                                 const EmbeddingError *__restrict__ emb_err, LogitCand *__restrict__ out_cands,
                                 uint32_t *__restrict__ out_cands_nan, MtpPick *__restrict__ out_picks) {
    __shared__ float s_val[kDraft2SMaxCap];
    __shared__ int32_t s_id[kDraft2SMaxCap];
    __shared__ float w_best_v[8];
    __shared__ int32_t w_best_id[8];
    __shared__ int32_t w_best_pos[8];
    __shared__ float w_sum[8];
    __shared__ float s_max_v;
    __shared__ float s_second_v;
    __shared__ float s_sum_exp;
    __shared__ int32_t s_best_id;

    const int t = blockIdx.x;
    const uint32_t has_nan = row_nan[t] ? 1u : 0u;
    const int n = min(cand_cnt[t], cap);

    if (has_nan || n <= 0) {
        if (out_cands != nullptr && threadIdx.x < kLogitCands) {
            out_cands[(int64_t)t * kLogitCands + threadIdx.x] = {-INFINITY, INT32_MAX};
        }
        if (threadIdx.x == 0) {
            if (out_cands_nan != nullptr) out_cands_nan[t] = has_nan;
            if (out_picks != nullptr) {
                MtpPick &pk = out_picks[t];
                pk.best = 0;
                pk.best_v = -INFINITY;
                pk.second_v = -INFINITY;
                pk.nan = has_nan;
                pk.p_top1 = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    pk.err[i] = router_err[i];
                    pk.err[3 + i] = expert_err[i];
                    pk.err[6 + i] = attn_err[i];
                }
                pk.emb = *emb_err;
            }
        }
        return;
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_val[i] = vals[(int64_t)t * cap + i];
        s_id[i] = orig_ids[(int64_t)t * cap + i];
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int nw = blockDim.x / warpSize;

    // 1. Find global max_v (for MtpPick softmax stability; also top0 when we only need p_top1)
    if (out_picks != nullptr) {
        float lmax = -INFINITY;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            lmax = fmaxf(lmax, s_val[i]);
        }
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            lmax = fmaxf(lmax, __shfl_xor(lmax, off));
        }
        if (lane == 0) w_best_v[wave] = lmax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float gm = w_best_v[0];
            for (int w = 1; w < nw; ++w) gm = fmaxf(gm, w_best_v[w]);
            s_max_v = gm;
        }
        __syncthreads();

        const float max_v = s_max_v;
        const float inv_T = inv_T_dev[t];
        float lsum = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = s_val[i];
            const float diff = (v == max_v) ? 0.0f : (v - max_v) * inv_T;
            lsum += isfinite(diff) ? __expf(diff) : 0.0f;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            lsum += __shfl_xor(lsum, off);
        }
        if (lane == 0) w_sum[wave] = lsum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float gs = 0.0f;
            for (int w = 0; w < nw; ++w) gs += w_sum[w];
            s_sum_exp = gs;
            s_second_v = -INFINITY;
        }
        __syncthreads();
    }

    // 2. Select exact top-k (k = kLogitCands if out_cands != nullptr, else just top-2 for MtpPick)
    const int k_want = (out_cands != nullptr) ? kLogitCands : 2;
    const int k_iters = min(k_want, n);
    for (int r = 0; r < k_iters; ++r) {
        float bv = -INFINITY;
        int32_t bid = INT32_MAX;
        int32_t bpos = -1;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = s_val[i];
            const int32_t id = s_id[i];
            if (v > bv || (v == bv && id < bid)) {
                bv = v;
                bid = id;
                bpos = i;
            }
        }
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off);
            const int32_t oid = __shfl_xor(bid, off);
            const int32_t opos = __shfl_xor(bpos, off);
            if (ov > bv || (ov == bv && oid < bid)) {
                bv = ov;
                bid = oid;
                bpos = opos;
            }
        }
        if (lane == 0) {
            w_best_v[wave] = bv;
            w_best_id[wave] = bid;
            w_best_pos[wave] = bpos;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float fv = w_best_v[0];
            int32_t fid = w_best_id[0];
            int32_t fpos = w_best_pos[0];
            for (int w = 1; w < nw; ++w) {
                if (w_best_v[w] > fv || (w_best_v[w] == fv && w_best_id[w] < fid)) {
                    fv = w_best_v[w];
                    fid = w_best_id[w];
                    fpos = w_best_pos[w];
                }
            }
            if (out_cands != nullptr) {
                out_cands[(int64_t)t * kLogitCands + r] = {fv, fid};
            }
            if (r == 0) {
                s_best_id = fid;
                s_max_v = fv;
            } else if (r == 1) {
                s_second_v = fv;
            }
            if (fpos >= 0) s_val[fpos] = -INFINITY;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (out_cands != nullptr) {
            for (int r = k_iters; r < kLogitCands; ++r) {
                out_cands[(int64_t)t * kLogitCands + r] = {-INFINITY, INT32_MAX};
            }
        }
        if (out_cands_nan != nullptr) {
            out_cands_nan[t] = 0u;
        }
        if (out_picks != nullptr) {
            MtpPick &pk = out_picks[t];
            pk.best = s_best_id;
            pk.best_v = s_max_v;
            pk.second_v = s_second_v;
            pk.nan = 0u;
            pk.p_top1 = (s_sum_exp > 0.0f && isfinite(s_sum_exp)) ? (1.0f / s_sum_exp) : 0.0f;
            for (int i = 0; i < 3; ++i) {
                pk.err[i] = router_err[i];
                pk.err[3 + i] = expert_err[i];
                pk.err[6 + i] = attn_err[i];
            }
            pk.emb = *emb_err;
        }
    }
}

struct WsLayout {
    size_t off_xr = 0;
    size_t off_preview = 0;
    size_t off_hist1 = 0;
    size_t off_hist2 = 0;
    size_t off_thr1 = 0;
    size_t off_need2 = 0;
    size_t off_gt1 = 0;
    size_t off_thr24 = 0;
    size_t off_gt_tot = 0;
    size_t off_cand_cnt = 0;
    size_t off_cnt_gt = 0;
    size_t off_cnt_eq = 0;
    size_t off_row_nan = 0;
    size_t off_cand_idx = 0;
    size_t off_exact_vals = 0;
    size_t off_exact_ids = 0;
    size_t total = 0;
};

WsLayout compute_ws_layout(int64_t T, int64_t V, int64_t R, int32_t cap) {
    WsLayout l;
    size_t cur = 0;
    auto alloc = [&](size_t bytes) {
        size_t o = align_up(cur);
        cur = o + bytes;
        return o;
    };
    l.off_xr = alloc((size_t)(T * R) * sizeof(uint16_t));
    l.off_preview = alloc((size_t)(T * V) * sizeof(float));
    l.off_hist1 = alloc((size_t)(T * kBins) * sizeof(uint32_t));
    l.off_hist2 = alloc((size_t)(T * kBins) * sizeof(uint32_t));
    l.off_thr1 = alloc((size_t)T * sizeof(int32_t));
    l.off_need2 = alloc((size_t)T * sizeof(int32_t));
    l.off_gt1 = alloc((size_t)T * sizeof(int32_t));
    l.off_thr24 = alloc((size_t)T * sizeof(int32_t));
    l.off_gt_tot = alloc((size_t)T * sizeof(int32_t));
    l.off_cand_cnt = alloc((size_t)T * sizeof(int32_t));
    l.off_cnt_gt = alloc((size_t)T * sizeof(int32_t));
    l.off_cnt_eq = alloc((size_t)T * sizeof(int32_t));
    l.off_row_nan = alloc((size_t)T * sizeof(uint32_t));
    l.off_cand_idx = alloc((size_t)(T * cap) * sizeof(int32_t));
    l.off_exact_vals = alloc((size_t)(T * cap) * sizeof(float));
    l.off_exact_ids = alloc((size_t)(T * cap) * sizeof(int32_t));
    l.total = align_up(cur);
    return l;
}

}  // namespace

size_t draft2s_workspace_bytes(int64_t max_T, int64_t max_V, int64_t max_R, int32_t max_cap) {
    STRIX_CHECK(max_T >= 1 && max_V >= 1 && max_R >= 1 && max_cap >= 1 && max_cap <= kDraft2SMaxCap,
                "draft2s_workspace_bytes: invalid dims T=", max_T, ", V=", max_V, ", R=", max_R, ", cap=", max_cap);
    return compute_ws_layout(max_T, max_V, max_R, max_cap).total;
}

void draft2s_forward(const void *x_bf16, const Draft2SParams &p, int64_t T, int64_t n_valid, const float *inv_T_dev,
                     const uint32_t *router_err, const uint32_t *expert_err, const uint32_t *attn_err,
                     const EmbeddingError *emb_err, LogitCand *out_cands, uint32_t *out_cands_nan, MtpPick *out_picks,
                     void *ws, size_t ws_bytes, hipStream_t stream) {
    const char *fn = "draft2s_forward";
    STRIX_CHECK(x_bf16 && p.E_bf16 && p.P_q4.q && p.W_q8.q && ws, fn, ": null pointer");
    STRIX_CHECK(T >= 1 && p.V >= 1 && n_valid >= 1 && n_valid <= p.V, fn, ": bad T=", T, ", n_valid=", n_valid,
                ", V=", p.V);
    STRIX_CHECK(p.R >= 32 && p.R == p.P_q4.K && p.P_q4.N == p.V, fn, ": P_q4 mismatch");
    STRIX_CHECK(p.W_q8.K % 16 == 0 && p.W_q8.K % p.W_q8.G == 0, fn, ": W_q8 K=", p.W_q8.K, " G=", p.W_q8.G);
    STRIX_CHECK(p.M_cand >= 1 && p.cap >= p.M_cand && p.cap <= kDraft2SMaxCap, fn, ": bad M_cand=", p.M_cand,
                ", cap=", p.cap);
    STRIX_CHECK(out_cands != nullptr || out_picks != nullptr, fn, ": at least one output required");
    if (out_picks != nullptr) {
        STRIX_CHECK(inv_T_dev && router_err && expert_err && attn_err && emb_err, fn,
                    ": inv_T_dev and error slots required for out_picks");
    }

    const WsLayout l = compute_ws_layout(T, p.V, p.R, p.cap);
    STRIX_CHECK(ws_bytes >= l.total, fn, ": workspace ", ws_bytes, " < required ", l.total);

    auto *base = static_cast<uint8_t *>(ws);
    auto ptr = [&](size_t off) -> void * { return base + off; };
    auto *xr = static_cast<uint16_t *>(ptr(l.off_xr));
    auto *preview = static_cast<float *>(ptr(l.off_preview));
    auto *hist1 = static_cast<uint32_t *>(ptr(l.off_hist1));
    auto *hist2 = static_cast<uint32_t *>(ptr(l.off_hist2));
    auto *thr1 = static_cast<int32_t *>(ptr(l.off_thr1));
    auto *need2 = static_cast<int32_t *>(ptr(l.off_need2));
    auto *gt1 = static_cast<int32_t *>(ptr(l.off_gt1));
    auto *thr24 = static_cast<int32_t *>(ptr(l.off_thr24));
    auto *gt_tot = static_cast<int32_t *>(ptr(l.off_gt_tot));
    auto *cand_cnt = static_cast<int32_t *>(ptr(l.off_cand_cnt));
    auto *cnt_gt = static_cast<int32_t *>(ptr(l.off_cnt_gt));
    auto *cnt_eq = static_cast<int32_t *>(ptr(l.off_cnt_eq));
    auto *row_nan = static_cast<uint32_t *>(ptr(l.off_row_nan));
    auto *cand_idx = static_cast<int32_t *>(ptr(l.off_cand_idx));
    auto *exact_vals = static_cast<float *>(ptr(l.off_exact_vals));
    auto *exact_ids = static_cast<int32_t *>(ptr(l.off_exact_ids));

    // Stage 1: low-rank Act-PCA projection xr = x @ E^T
    linear_bf16w(x_bf16, p.E_bf16, xr, T, p.R, p.W_q8.K, Act::BF16, Act::BF16, stream);

    // Stage 2: Q4 preview scores + exact zero-overflow 24-bit radix selection
    linear_q4(xr, p.P_q4, preview, T, Act::BF16, Act::F32, stream);

    const unsigned zg = (unsigned)std::min<int64_t>(256, std::max<int64_t>(1, (T * kBins + kThreads - 1) / kThreads));
    preview_zero_kernel<<<zg, kThreads, 0, stream>>>(hist1, hist2, cnt_gt, cnt_eq, row_nan, T);

    const dim3 hgrid((unsigned)std::min<int64_t>(256, std::max<int64_t>(1, (n_valid + 1023) / 1024)), (unsigned)T);
    const int32_t eff_M = (int32_t)std::min<int64_t>(p.M_cand, n_valid);
    preview_hist1_kernel<<<hgrid, kThreads, 0, stream>>>(preview, p.V, n_valid, hist1, row_nan);
    preview_thresh1_kernel<<<(unsigned)T, kThreads, 0, stream>>>(hist1, eff_M, thr1, need2, gt1);
    preview_hist2_kernel<<<hgrid, kThreads, 0, stream>>>(preview, p.V, n_valid, thr1, hist2);
    preview_thresh2_kernel<<<(unsigned)T, kThreads, 0, stream>>>(hist2, thr1, need2, gt1, p.cap, thr24, gt_tot, cand_cnt);
    preview_compact_kernel<<<hgrid, kThreads, 0, stream>>>(preview, p.V, n_valid, thr24, gt_tot, p.cap, cand_idx, cnt_gt,
                                                           cnt_eq);

    // Stage 3: exact Q8 dot products on gathered candidates + fused top-kLogitCands / MtpPick
    q8_gather_rows_kernel<<<dim3((unsigned)p.cap, (unsigned)T), kGatherThreads, 0, stream>>>(
        static_cast<const uint16_t *>(x_bf16), p.W_q8, p.row_map, cand_idx, cand_cnt, p.cap, row_nan, row_nan,
        exact_vals, exact_ids);

    gather_top20_and_pick_kernel<<<(unsigned)T, kThreads, 0, stream>>>(
        exact_vals, exact_ids, cand_cnt, row_nan, p.cap, inv_T_dev, router_err, expert_err, attn_err, emb_err,
        out_cands, out_cands_nan, out_picks);

    STRIX_HIP_CHECK(hipGetLastError(), fn, ": kernel launch");
}

}  // namespace strix::kernels
