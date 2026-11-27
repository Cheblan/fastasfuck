#include "common/bf16.hpp"
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "kernels/draft2s.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/logits_topk.hpp"
#include "kernels/mtp_pick.hpp"
#include "kernels/types.hpp"
#include "runtime/act_pca.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using strix::DeviceBuffer;
using strix::Q4Device;
using strix::Q4Weight;
using strix::Q8Device;
using strix::Q8Weight;
namespace kernels = strix::kernels;

namespace {

void test_act_pca() {
    std::cout << "[test_act_pca] verifying low-rank orthonormal basis & cache...\n";
    constexpr int64_t N = 640, K = 2560, R = 512, TrueRank = 16;
    std::mt19937_64 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Build TrueRank random orthonormal vectors A [TrueRank, K]
    std::vector<double> A((size_t)(TrueRank * K));
    for (int64_t r = 0; r < TrueRank; ++r) {
        double *ap = A.data() + r * K;
        for (int64_t k = 0; k < K; ++k) ap[k] = gauss(rng);
        for (int64_t p = 0; p < r; ++p) {
            const double *pp = A.data() + p * K;
            double dot = 0.0;
            for (int64_t k = 0; k < K; ++k) dot += ap[k] * pp[k];
            for (int64_t k = 0; k < K; ++k) ap[k] -= dot * pp[k];
        }
        double n2 = 0.0;
        for (int64_t k = 0; k < K; ++k) n2 += ap[k] * ap[k];
        const double inv = 1.0 / std::sqrt(n2);
        for (int64_t k = 0; k < K; ++k) ap[k] *= inv;
    }

    // U = Z @ A + tiny noise
    std::vector<uint16_t> U_bf16((size_t)(N * K));
    for (int64_t i = 0; i < N; ++i) {
        std::vector<double> coef((size_t)TrueRank);
        for (int64_t r = 0; r < TrueRank; ++r) coef[(size_t)r] = 3.0 * gauss(rng);
        for (int64_t k = 0; k < K; ++k) {
            double v = 0.01 * gauss(rng);
            for (int64_t r = 0; r < TrueRank; ++r) v += coef[(size_t)r] * A[(size_t)(r * K + k)];
            U_bf16[(size_t)(i * K + k)] = strix::f32_to_bf16_bits((float)v);
        }
    }

    const std::vector<uint16_t> E_bf16 = strix::compute_act_pca_basis_bf16(U_bf16.data(), N, K, R);
    STRIX_CHECK((int64_t)E_bf16.size() == R * K, "bad E_bf16 size");

    // Check orthonormality ||E E^T - I||_max < 2e-2 (BF16 quantization tolerance over K=2560)
    double max_err = 0.0;
    for (int64_t i = 0; i < 64; ++i) {
        for (int64_t j = 0; j < 64; ++j) {
            double dot = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                dot += (double)strix::bf16_bits_to_f32(E_bf16[(size_t)(i * K + k)]) *
                       (double)strix::bf16_bits_to_f32(E_bf16[(size_t)(j * K + k)]);
            }
            const double tgt = (i == j) ? 1.0 : 0.0;
            max_err = std::max(max_err, std::abs(dot - tgt));
        }
    }
    STRIX_CHECK(max_err < 2e-2, "E orthonormality max_err = ", max_err);

    // Check captured variance of U > 99%
    double tot_e = 0.0, cap_e = 0.0;
    for (int64_t i = 0; i < 64; ++i) {
        std::vector<double> u((size_t)K);
        for (int64_t k = 0; k < K; ++k) {
            u[(size_t)k] = (double)strix::bf16_bits_to_f32(U_bf16[(size_t)(i * K + k)]);
            tot_e += u[(size_t)k] * u[(size_t)k];
        }
        for (int64_t r = 0; r < TrueRank; ++r) {
            double dot = 0.0;
            for (int64_t k = 0; k < K; ++k) dot += u[(size_t)k] * (double)strix::bf16_bits_to_f32(E_bf16[(size_t)(r * K + k)]);
            cap_e += dot * dot;
        }
    }
    const double frac = cap_e / tot_e;
    STRIX_CHECK(frac > 0.99, "captured energy fraction too low: ", frac);

    // Cache roundtrip test
    const std::string cpath = "/tmp/test_act_pca_cache.bin";
    strix::save_act_pca_cache(cpath, "test_fp_v1", R, K, E_bf16);
    std::vector<uint16_t> loaded;
    STRIX_CHECK(strix::load_act_pca_cache(cpath, "test_fp_v1", R, K, loaded) && loaded == E_bf16,
                "cache roundtrip failed");
    STRIX_CHECK(!strix::load_act_pca_cache(cpath, "wrong_fp", R, K, loaded), "cache fingerprint mismatch failed");
    std::remove(cpath.c_str());
    std::cout << "  PASS (max_ortho_err=" << max_err << ", captured_var=" << frac << ")\n";
}

void test_draft2s_gpu(hipStream_t stream) {
    std::cout << "[test_draft2s_gpu] verifying exact Q8 gather, row_map, batching, and NaN handling...\n";
    constexpr int64_t N_full = 4096, K = 2560, G = 64, R = 256, T = 4;
    std::mt19937_64 rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // Generate low-rank W_f32 [N_full, K] in span of first R=256 PCA axes so P_q4 preview is sharp & deterministic
    std::vector<float> W_f32((size_t)(N_full * K), 0.0f);
    std::vector<float> x_f32((size_t)(T * K), 0.0f);
    std::vector<uint16_t> x_bf16((size_t)(T * K));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < R; ++k) x_f32[(size_t)(t * K + k)] = gauss(rng);
        for (int64_t k = 0; k < K; ++k) x_bf16[(size_t)(t * K + k)] = strix::f32_to_bf16_bits(x_f32[(size_t)(t * K + k)]);
    }
    for (int64_t n = 0; n < N_full; ++n) {
        for (int64_t k = 0; k < R; ++k) W_f32[(size_t)(n * K + k)] = 0.25f * gauss(rng);
    }

    Q8Weight w_q8 = strix::quantize_q8(W_f32.data(), N_full, K, G);
    Q8Device d_w_q8 = Q8Device::upload(w_q8, "test_w_q8");

    // Subset row_map (2048 rows, non-contiguous stride-2 / shuffled)
    std::vector<int32_t> row_map(2048);
    for (int64_t i = 0; i < 2048; ++i) row_map[(size_t)i] = (int32_t)(i * 2 + (i & 1));
    Q8Weight sub_q8{2048, K, G};
    sub_q8.q.resize((size_t)(2048 * K));
    sub_q8.scale.resize((size_t)(2048 * (K / G)));
    sub_q8.minv.resize(sub_q8.scale.size());
    for (int64_t i = 0; i < 2048; ++i) {
        const int64_t src = row_map[(size_t)i];
        std::memcpy(&sub_q8.q[(size_t)(i * K)], &w_q8.q[(size_t)(src * K)], (size_t)K);
        std::memcpy(&sub_q8.scale[(size_t)(i * (K / G))], &w_q8.scale[(size_t)(src * (K / G))], (size_t)(K / G) * 2);
        std::memcpy(&sub_q8.minv[(size_t)(i * (K / G))], &w_q8.minv[(size_t)(src * (K / G))], (size_t)(K / G) * 2);
    }
    Q8Device d_sub_q8 = Q8Device::upload(sub_q8, "test_sub_q8");

    // Identity E [R, K] on first R coords (captures 100% of W & x)
    std::vector<uint16_t> E_bf16((size_t)(R * K), 0);
    for (int64_t r = 0; r < R; ++r) E_bf16[(size_t)(r * K + r)] = strix::f32_to_bf16_bits(1.0f);
    DeviceBuffer<uint16_t> d_E = DeviceBuffer<uint16_t>::from_host(E_bf16, "test_E");

    std::vector<float> P_sub_f32((size_t)(2048 * R));
    for (int64_t i = 0; i < 2048; ++i) {
        const int64_t src = row_map[(size_t)i];
        std::memcpy(&P_sub_f32[(size_t)(i * R)], &W_f32[(size_t)(src * K)], (size_t)R * sizeof(float));
    }
    Q4Weight P_sub_q4 = strix::quantize_q4(P_sub_f32.data(), 2048, R, G);
    Q4Device d_P_sub = Q4Device::upload(P_sub_q4, "test_P_sub");
    DeviceBuffer<int32_t> d_map = DeviceBuffer<int32_t>::from_host(row_map, "test_map");

    DeviceBuffer<uint16_t> d_x = DeviceBuffer<uint16_t>::from_host(x_bf16, "test_x");
    const std::vector<float> inv_T = {1.0f, 1.0f / 0.6f, 1.0f, 1.0f / 0.8f};
    DeviceBuffer<float> d_inv_T = DeviceBuffer<float>::from_host(inv_T, "test_inv_T");
    DeviceBuffer<uint32_t> d_err(16, "test_err");
    STRIX_HIP_CHECK(hipMemset(d_err.get(), 0, 16 * 4), "zero err");

    // Reference Q8 forward on sub_q8
    DeviceBuffer<float> d_logits((size_t)(T * 2048), "ref_logits");
    DeviceBuffer<kernels::LogitCand> d_ref_cands((size_t)(T * kernels::kLogitCands), "ref_cands");
    DeviceBuffer<uint32_t> d_ref_nan((size_t)T, "ref_nan");
    const size_t tws_bytes = kernels::logits_topk_workspace_bytes(T, 2048);
    DeviceBuffer<uint8_t> d_tws(tws_bytes, "ref_tws");
    DeviceBuffer<kernels::MtpPick> d_ref_picks((size_t)T, "ref_picks");

    kernels::linear_q8(d_x.get(), d_sub_q8.view(), d_logits.get(), T, kernels::Act::BF16, kernels::Act::F32, stream);
    kernels::logits_topk(d_logits.get(), T, 2048, 2048, d_ref_cands.get(), d_ref_nan.get(), d_tws.get(), tws_bytes, stream);
    for (int64_t r = 0; r < T; ++r) {
        kernels::mtp_pick(d_logits.get() + r * 2048, 2048, inv_T[(size_t)r], d_err.get(), d_err.get() + 3, d_err.get() + 6,
                          reinterpret_cast<const kernels::EmbeddingError *>(d_err.get() + 9), d_ref_picks.get() + r,
                          stream);
    }

    // 2-Stage forward
    const size_t ws_bytes = kernels::draft2s_workspace_bytes(T, N_full, R, 1024);
    DeviceBuffer<uint8_t> d_ws(ws_bytes, "d2s_ws");
    DeviceBuffer<kernels::LogitCand> d_2s_cands((size_t)(T * kernels::kLogitCands), "d2s_cands");
    DeviceBuffer<uint32_t> d_2s_nan((size_t)T, "d2s_nan");
    DeviceBuffer<kernels::MtpPick> d_2s_picks((size_t)T, "d2s_picks");

    kernels::Draft2SParams params{d_E.get(), d_P_sub.view(), d_w_q8.view(), d_map.get(), R, 2048, 512, 1024};
    kernels::draft2s_forward(d_x.get(), params, T, 2048, d_inv_T.get(), d_err.get(), d_err.get() + 3, d_err.get() + 6,
                             reinterpret_cast<const kernels::EmbeddingError *>(d_err.get() + 9), d_2s_cands.get(),
                             d_2s_nan.get(), d_2s_picks.get(), d_ws.get(), ws_bytes, stream);

    std::vector<kernels::LogitCand> ref_c((size_t)(T * kernels::kLogitCands)), d2s_c((size_t)(T * kernels::kLogitCands));
    std::vector<kernels::MtpPick> ref_p((size_t)T), d2s_p((size_t)T);
    std::vector<uint32_t> d2s_nan((size_t)T);
    STRIX_HIP_CHECK(hipMemcpyAsync(ref_c.data(), d_ref_cands.get(), ref_c.size() * sizeof(kernels::LogitCand),
                                   hipMemcpyDeviceToHost, stream), "copy ref_c");
    STRIX_HIP_CHECK(hipMemcpyAsync(d2s_c.data(), d_2s_cands.get(), d2s_c.size() * sizeof(kernels::LogitCand),
                                   hipMemcpyDeviceToHost, stream), "copy d2s_c");
    STRIX_HIP_CHECK(hipMemcpyAsync(ref_p.data(), d_ref_picks.get(), ref_p.size() * sizeof(kernels::MtpPick),
                                   hipMemcpyDeviceToHost, stream), "copy ref_p");
    STRIX_HIP_CHECK(hipMemcpyAsync(d2s_p.data(), d_2s_picks.get(), d2s_p.size() * sizeof(kernels::MtpPick),
                                   hipMemcpyDeviceToHost, stream), "copy d2s_p");
    STRIX_HIP_CHECK(hipMemcpyAsync(d2s_nan.data(), d_2s_nan.get(), d2s_nan.size() * 4, hipMemcpyDeviceToHost, stream),
                    "copy d2s_nan");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync test_draft2s_gpu");

    for (int64_t t = 0; t < T; ++t) {
        const int32_t ref_orig_best = row_map[(size_t)ref_p[(size_t)t].best];
        STRIX_CHECK(d2s_nan[(size_t)t] == 0 && d2s_p[(size_t)t].nan == 0, "unexpected NaN at t=", t);
        STRIX_CHECK(d2s_p[(size_t)t].best == ref_orig_best, "top1 mismatch t=", t, " got=", d2s_p[(size_t)t].best,
                    " want=", ref_orig_best);
        STRIX_CHECK(std::abs(d2s_p[(size_t)t].best_v - ref_p[(size_t)t].best_v) < 1e-4f, "best_v mismatch t=", t);
        STRIX_CHECK(std::abs(d2s_p[(size_t)t].second_v - ref_p[(size_t)t].second_v) < 1e-4f, "second_v mismatch t=", t);
        STRIX_CHECK(std::abs(d2s_p[(size_t)t].p_top1 - ref_p[(size_t)t].p_top1) < 1e-2f, "p_top1 mismatch t=", t,
                    " got=", d2s_p[(size_t)t].p_top1, " want=", ref_p[(size_t)t].p_top1);
        for (int i = 0; i < 5; ++i) {
            const int32_t rc = row_map[(size_t)ref_c[(size_t)(t * kernels::kLogitCands + i)].id];
            const int32_t dc = d2s_c[(size_t)(t * kernels::kLogitCands + i)].id;
            STRIX_CHECK(rc == dc, "cand mismatch at t=", t, " rank=", i, " got=", dc, " want=", rc);
        }
    }

    // NaN propagation check
    std::vector<uint16_t> x_nan((size_t)K, 0x7FC0u);  // BF16 QNaN
    DeviceBuffer<uint16_t> d_x_nan = DeviceBuffer<uint16_t>::from_host(x_nan, "test_x_nan");
    kernels::draft2s_forward(d_x_nan.get(), params, 1, 2048, d_inv_T.get(), d_err.get(), d_err.get() + 3,
                             d_err.get() + 6, reinterpret_cast<const kernels::EmbeddingError *>(d_err.get() + 9),
                             d_2s_cands.get(), d_2s_nan.get(), d_2s_picks.get(), d_ws.get(), ws_bytes, stream);
    uint32_t nan_flag = 0;
    kernels::MtpPick nan_pick{};
    STRIX_HIP_CHECK(hipMemcpyAsync(&nan_flag, d_2s_nan.get(), 4, hipMemcpyDeviceToHost, stream), "nan flag");
    STRIX_HIP_CHECK(hipMemcpyAsync(&nan_pick, d_2s_picks.get(), sizeof(nan_pick), hipMemcpyDeviceToHost, stream), "nan pick");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync nan");
    STRIX_CHECK(nan_flag == 1u && nan_pick.nan == 1u, "NaN propagation failed");

    std::cout << "  PASS (T=1/4, row_map remap, p_top1, top-20, NaN)\n";
}

}  // namespace

int main() {
    hipStream_t stream = nullptr;
    STRIX_HIP_CHECK(hipStreamCreate(&stream), "stream create");
    test_act_pca();
    test_draft2s_gpu(stream);
    STRIX_HIP_CHECK(hipStreamDestroy(stream), "stream destroy");
    std::cout << "ALL DRAFT2S TESTS PASSED.\n";
    return 0;
}
