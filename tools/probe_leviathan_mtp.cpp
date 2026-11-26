// Diagnostic probe for Leviathan Speculative Sampling vs Greedy Argmax MTP at high temperatures.
// Compares empirical acceptance rates, expected token yield E[accepted], TVD overlap sum(min(p, q)),
// draft temperature scaling, compact en+ru vocab filtering, and GPU logits_topk vs mtp_pick latency.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/logits_topk.hpp"
#include "kernels/mtp_pick.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vocab_filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace strix;

namespace {

struct ProbDist {
    std::vector<int32_t> ids;
    std::vector<double> probs;  // sums to 1.0
    int32_t top1_id = -1;
    double p_top1 = 0.0;
};

// Extract exact top_k + top_p truncated probability distribution (matching Sampler::draw_from_candidates).
ProbDist build_top_dist(const float *logits, int64_t n_valid, double temp, int64_t top_k, double top_p,
                        const std::vector<uint8_t> *allowed = nullptr) {
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    const size_t k = (size_t)std::min<int64_t>(n_valid, std::max<int64_t>(1, top_k));
    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve(k + 1);
    for (int64_t i = 0; i < n_valid; ++i) {
        if (allowed && !(*allowed)[(size_t)i]) continue;
        const float v = logits[i];
        if (!std::isfinite(v)) continue;
        if (cand.size() == k) {
            if (!(v > cand.back().first)) continue;
            cand.pop_back();
        }
        const std::pair<float, int32_t> c{v, (int32_t)i};
        cand.insert(std::upper_bound(cand.begin(), cand.end(), c, better), c);
    }
    STRIX_CHECK(!cand.empty(), "build_top_dist: empty candidates");

    ProbDist out;
    out.top1_id = cand[0].second;
    if (temp <= 0.0) {
        out.ids = {cand[0].second};
        out.probs = {1.0};
        out.p_top1 = 1.0;
        return out;
    }

    const size_t n = cand.size();
    std::vector<double> raw(n);
    const double max_s = (double)cand[0].first / temp;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        raw[i] = std::exp((double)cand[i].first / temp - max_s);
        sum += raw[i];
    }
    size_t keep = n;
    if (top_p < 1.0) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            acc += raw[i] / sum;
            if (acc >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) kept_sum += raw[i];
    out.ids.resize(keep);
    out.probs.resize(keep);
    for (size_t i = 0; i < keep; ++i) {
        out.ids[i] = cand[i].second;
        out.probs[i] = raw[i] / kept_sum;
    }
    out.p_top1 = out.probs[0];
    return out;
}

double prob_of(const ProbDist &d, int32_t id) {
    for (size_t i = 0; i < d.ids.size(); ++i) {
        if (d.ids[i] == id) return d.probs[i];
    }
    return 0.0;
}

double overlap_tvd(const ProbDist &p, const ProbDist &q) {
    double s = 0.0;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        const double qi = prob_of(q, p.ids[i]);
        if (qi > 0.0) s += std::min(p.probs[i], qi);
    }
    return s;
}

int32_t sample_dist(const ProbDist &d, std::mt19937_64 &rng) {
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < d.ids.size(); ++i) {
        acc += d.probs[i];
        if (u < acc) return d.ids[i];
    }
    return d.ids.back();
}

int32_t sample_residual(const ProbDist &p, const ProbDist &q, std::mt19937_64 &rng) {
    std::vector<double> r(p.ids.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        const double diff = std::max(0.0, p.probs[i] - prob_of(q, p.ids[i]));
        r[i] = diff;
        sum += diff;
    }
    if (sum <= 1e-15) return sample_dist(p, rng);
    const double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
    double acc = 0.0;
    for (size_t i = 0; i < p.ids.size(); ++i) {
        acc += r[i];
        if (u < acc) return p.ids[i];
    }
    return p.ids.back();
}

// Standalone cost estimator using measured solo_verify_ms_grid on Thinking-Cap Qwen3.8-Flash-Next.
double solo_verify_cost_ms(int64_t rows) {
    static constexpr double grid[] = {
        0.0, 25.27, 30.80, 36.04, 41.08, 46.43, 48.07, 53.08, 56.90
    };
    if (rows <= 0) return 0.0;
    if (rows <= 8) return grid[rows];
    return grid[8] + (double)(rows - 8) * 4.5;
}

struct ModeConfig {
    std::string name;
    bool leviathan = false;
    double temp_target = 1.0;
    double temp_draft_scale = 1.0;
    double draft_top_p = 0.95;
    int64_t draft_top_k = 20;
    bool use_en_ru_mask = false;
    int k_steps = 5;
};

struct ModeStats {
    int64_t rounds = 0;
    int64_t drafted = 0;
    int64_t accepted = 0;
    int64_t emitted = 0;
    int64_t rej0 = 0;
    double sum_step0_argmax_p = 0.0;
    double sum_step0_overlap = 0.0;
    int64_t step0_n = 0;
    double est_time_ms = 0.0;
    std::vector<int64_t> step_trials = std::vector<int64_t>(8, 0);
    std::vector<int64_t> step_hits = std::vector<int64_t>(8, 0);
};

std::string wrap_chat(const std::string &user_text) {
    return "<|im_start|>user\n" + user_text + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n";
}

ModeStats run_mode(Qwen4ExpSession &session, const Tokenizer &tok,
                   const std::vector<std::vector<int32_t>> &prompt_tokens,
                   const std::vector<uint8_t> &allowed_en_ru,
                   const ModeConfig &cfg, int max_gen_tokens, int32_t eos_id, int32_t eot_id) {
    const int64_t n_valid = tok.size();
    ModeStats st;
    const double t_draft = cfg.temp_target * cfg.temp_draft_scale;

    for (size_t p_idx = 0; p_idx < prompt_tokens.size(); ++p_idx) {
        session.reset();
        std::mt19937_64 rng(42 + p_idx * 1009ULL);
        const std::vector<int32_t> &ptok = prompt_tokens[p_idx];
        std::vector<float> logits = session.forward(ptok, 1);

        int emitted_here = 0;
        int32_t carry = -1;
        while (emitted_here < max_gen_tokens) {
            int32_t id = -1;
            if (carry >= 0) {
                id = carry;
                carry = -1;
            } else {
                const ProbDist p0 = build_top_dist(logits.data(), n_valid, cfg.temp_target, 20, 0.95);
                id = sample_dist(p0, rng);
                ++emitted_here;
                ++st.emitted;
                if (id == eos_id || id == eot_id || emitted_here >= max_gen_tokens) break;
            }

            const int k = std::min<int>(cfg.k_steps, max_gen_tokens - emitted_here);
            if (k <= 0) break;

            std::vector<int32_t> ids{id};
            std::vector<ProbDist> draft_dists;
            draft_dists.reserve((size_t)k);

            for (int step = 0; step < k; ++step) {
                std::vector<float> dlog = session.forward_mtp(ids.back(), step);
                ProbDist q = build_top_dist(
                    dlog.data(), n_valid,
                    cfg.leviathan ? t_draft : cfg.temp_target,
                    cfg.leviathan ? cfg.draft_top_k : 20,
                    cfg.leviathan ? cfg.draft_top_p : 0.95,
                    cfg.use_en_ru_mask ? &allowed_en_ru : nullptr);
                int32_t d = cfg.leviathan ? sample_dist(q, rng) : q.top1_id;
                ids.push_back(d);
                draft_dists.push_back(std::move(q));
                if (d == eos_id || d == eot_id) break;
            }

            const int k_run = (int)ids.size() - 1;
            std::vector<float> ml = session.forward_verify(ids, k_run + 1);
            const int64_t row_stride = (int64_t)ml.size() / (k_run + 1);

            ++st.rounds;
            st.drafted += k_run;
            st.est_time_ms += solo_verify_cost_ms(k_run + 1) + (double)k_run * 1.57;

            int j = 0;
            int32_t v = -1;
            bool ended = false;
            for (; j < k_run; ++j) {
                const float *row_j = ml.data() + (int64_t)j * row_stride;
                const ProbDist p = build_top_dist(row_j, n_valid, cfg.temp_target, 20, 0.95);
                const ProbDist &q = draft_dists[(size_t)j];
                if (j == 0) {
                    st.sum_step0_argmax_p += prob_of(p, q.top1_id);
                    st.sum_step0_overlap += overlap_tvd(p, q);
                    ++st.step0_n;
                }
                const int32_t d = ids[(size_t)j + 1];
                bool accept = false;
                if (cfg.leviathan) {
                    const double pd = prob_of(p, d);
                    const double qd = std::max(1e-18, prob_of(q, d));
                    const double acc_prob = std::min(1.0, pd / qd);
                    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                    if (u < acc_prob) {
                        accept = true;
                        v = d;
                    } else {
                        accept = false;
                        v = sample_residual(p, q, rng);
                    }
                } else {
                    v = sample_dist(p, rng);
                    accept = (v == d);
                }

                if (j < (int)st.step_trials.size()) ++st.step_trials[(size_t)j];
                if (!accept) break;
                if (j < (int)st.step_hits.size()) ++st.step_hits[(size_t)j];

                ++st.accepted;
                ++st.emitted;
                ++emitted_here;
                if (v == eos_id || v == eot_id || emitted_here >= max_gen_tokens) {
                    ended = true;
                    ++j;
                    break;
                }
            }

            if (j == 0) ++st.rej0;
            if (ended) {
                if (j < k_run) session.keep_verify_prefix(j + 1);
                else session.keep_verify();
                break;
            }
            if (j == k_run) {
                session.keep_verify();
                logits.assign(ml.begin() + (size_t)k_run * (size_t)row_stride,
                              ml.begin() + (size_t)(k_run + 1) * (size_t)row_stride);
                continue;
            }

            session.keep_verify_prefix(j + 1);
            ++st.emitted;
            ++emitted_here;
            if (v == eos_id || v == eot_id || emitted_here >= max_gen_tokens) break;
            carry = v;
        }
    }
    return st;
}

void bench_topk_vs_mtp_pick(int64_t n_valid) {
    hipStream_t stream = nullptr;
    STRIX_HIP_CHECK(hipStreamCreate(&stream), "create stream");

    std::vector<float> h_logits((size_t)n_valid);
    for (int64_t i = 0; i < n_valid; ++i) h_logits[(size_t)i] = std::sin((float)i * 0.01f) * 5.0f;

    DeviceBuffer<float> d_logits((size_t)n_valid, "bench logits");
    STRIX_HIP_CHECK(hipMemcpy(d_logits.get(), h_logits.data(), (size_t)n_valid * sizeof(float), hipMemcpyHostToDevice), "upload");

    DeviceBuffer<uint32_t> d_err(16, "err");
    DeviceBuffer<kernels::EmbeddingError> d_emb(1, "emb");
    STRIX_HIP_CHECK(hipMemset(d_err.get(), 0, 16 * sizeof(uint32_t)), "memset err");
    STRIX_HIP_CHECK(hipMemset(d_emb.get(), 0, sizeof(kernels::EmbeddingError)), "memset emb");

    DeviceBuffer<kernels::MtpPick> d_pick(1, "pick");
    PinnedHostBuffer h_pick(sizeof(kernels::MtpPick), "h_pick");

    const size_t ws_bytes = kernels::logits_topk_workspace_bytes(1, n_valid);
    DeviceBuffer<uint8_t> d_ws(ws_bytes, "topk ws");
    DeviceBuffer<kernels::LogitCand> d_cands(kernels::kLogitCands, "cands");
    DeviceBuffer<uint32_t> d_nan(1, "nan");
    PinnedHostBuffer h_cands(kernels::kLogitCands * sizeof(kernels::LogitCand) + sizeof(uint32_t), "h_cands");

    // Warmup
    for (int i = 0; i < 20; ++i) {
        kernels::mtp_pick(d_logits.get(), n_valid, 1.0f, d_err.get(), d_err.get() + 3, d_err.get() + 6, d_emb.get(), d_pick.get(), stream);
        kernels::logits_topk(d_logits.get(), 1, n_valid, n_valid, d_cands.get(), d_nan.get(), d_ws.get(), ws_bytes, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    const int iters = 200;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        kernels::mtp_pick(d_logits.get(), n_valid, 1.0f, d_err.get(), d_err.get() + 3, d_err.get() + 6, d_emb.get(), d_pick.get(), stream);
        STRIX_HIP_CHECK(hipMemcpyAsync(h_pick.get(), d_pick.get(), sizeof(kernels::MtpPick), hipMemcpyDeviceToHost, stream), "d2h pick");
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync pick");
    }
    double pick_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;

    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        kernels::logits_topk(d_logits.get(), 1, n_valid, n_valid, d_cands.get(), d_nan.get(), d_ws.get(), ws_bytes, stream);
        STRIX_HIP_CHECK(hipMemcpyAsync(h_cands.get(), d_cands.get(), kernels::kLogitCands * sizeof(kernels::LogitCand), hipMemcpyDeviceToHost, stream), "d2h cands");
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync topk");
    }
    double topk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / iters;

    std::cout << "\n[GPU Kernel Overhead @ n=" << n_valid << " (end-to-end launch + D2H + sync)]\n";
    std::cout << "  mtp_pick   (64B D2H): " << std::fixed << std::setprecision(4) << pick_ms << " ms\n";
    std::cout << "  logits_topk(160B D2H): " << std::fixed << std::setprecision(4) << topk_ms << " ms (delta = +" << (topk_ms - pick_ms) << " ms)\n";
    STRIX_HIP_CHECK(hipStreamDestroy(stream), "destroy stream");
}

}  // namespace

int main(int argc, char **argv) {
    std::string weights_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/weights.faf";
    std::string ngram_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table";
    std::string tokenizer_path = "/models/Thinking-Cap-Qwen-3.8-FN-faf/tokenizer.json";
    int max_gen = 150;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--ngram" && i + 1 < argc) ngram_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--max-gen" && i + 1 < argc) max_gen = std::stoi(argv[++i]);
    }

    Tokenizer tok(tokenizer_path);
    const int64_t n_valid = tok.size();
    const int32_t eos_id = tok.id_of("<|endoftext|>");
    const int32_t eot_id = tok.id_of("<|im_end|>");

    std::vector<std::string> prompts = {
        wrap_chat("Write a robust, production-ready asynchronous task execution DAG in Python 3.12 using asyncio with cycle detection and cancellation."),
        wrap_chat("State and prove the Riemann-Lebesgue Lemma for functions in L1(R) with full algebraic rigor."),
        wrap_chat("Проанализируйте понятие zero-copy Unified Memory Architecture на AMD Strix Halo и переведите на академический русский язык.")
    };
    std::vector<std::vector<int32_t>> prompt_tokens;
    for (const auto &s : prompts) prompt_tokens.push_back(tok.encode(s));

    // Build en+ru allowed vocab mask matching production mtp-vocab-filter = en+ru
    const std::vector<int32_t> en_ru_ids = vocab_filter::filter_vocabulary(
        tok, vocab_filter::Core | vocab_filter::Latin | vocab_filter::Cyrillic, n_valid);
    std::vector<uint8_t> allowed_en_ru((size_t)n_valid, 0);
    for (int32_t id : en_ru_ids) allowed_en_ru[(size_t)id] = 1;

    bench_topk_vs_mtp_pick(n_valid);
    bench_topk_vs_mtp_pick((int64_t)en_ru_ids.size());

    std::cout << "\nLoading model: " << weights_path << " ...\n";
    Qwen4ExpModel model(weights_path, ngram_path, kernels::Act::BF16, false, 8388608, 1.0f, WeightPlacement::Device);
    model.make_draft_head(model.dims().vocab, /*quantize_q4=*/true, /*G=*/64, /*threads=*/16);
    Qwen4ExpSession session(model, 4096, 1024, PrefillMath::WmmaBf16, /*mtp=*/true, KvFormat::Q8, 1);
    PrecisionOptions prec;
    prec.gdn = kernels::DrPrecision::HalfU;
    prec.attention = kernels::AttnPrecision::Fast;
    session.set_precision(prec);

    // Warmup forward
    session.forward({prompt_tokens[0][0]}, 1);
    session.reset();

    std::vector<ModeConfig> modes = {
        {"Argmax  T=1.0 (full)",   false, 1.0, 1.00, 0.95, 20, false, 5},
        {"Argmax  T=1.0 (en+ru)",  false, 1.0, 1.00, 0.95, 20, true,  5},
        {"Leviath T=1.0 Td=1.00",  true,  1.0, 1.00, 0.95, 20, true,  5},
        {"Leviath T=1.0 Td=0.85",  true,  1.0, 0.85, 0.95, 20, true,  5},
        {"Leviath T=1.0 Td=0.70",  true,  1.0, 0.70, 0.95, 20, true,  5},
        {"Leviath T=1.0 full_q",   true,  1.0, 1.00, 0.95, 20, false, 5},
        {"Argmax  T=0.6 (en+ru)",  false, 0.6, 1.00, 0.95, 20, true,  5},
        {"Leviath T=0.6 Td=1.00",  true,  0.6, 1.00, 0.95, 20, true,  5},
        {"Leviath T=0.6 Td=0.85",  true,  0.6, 0.85, 0.95, 20, true,  5},
    };

    std::cout << "\n========================================================================================================\n";
    std::cout << "Mode                    | Acc/Dr (%) | E(tok/rnd) | Rej0 (%) | S0 ArgmaxP | S0 SumMinPQ | Est tok/s | Step acc k=0..4 (%)\n";
    std::cout << "------------------------|------------|------------|----------|------------|-------------|-----------|--------------------------\n";
    for (const auto &cfg : modes) {
        ModeStats st = run_mode(session, tok, prompt_tokens, allowed_en_ru, cfg, max_gen, eos_id, eot_id);
        const double acc_pct = st.drafted > 0 ? 100.0 * (double)st.accepted / (double)st.drafted : 0.0;
        const double E = st.rounds > 0 ? (double)st.emitted / (double)st.rounds : 0.0;
        const double rej0_pct = st.rounds > 0 ? 100.0 * (double)st.rej0 / (double)st.rounds : 0.0;
        const double s0_arg = st.step0_n > 0 ? 100.0 * st.sum_step0_argmax_p / (double)st.step0_n : 0.0;
        const double s0_ovl = st.step0_n > 0 ? 100.0 * st.sum_step0_overlap / (double)st.step0_n : 0.0;
        const double est_tps = st.est_time_ms > 0 ? 1000.0 * (double)st.emitted / st.est_time_ms : 0.0;
        std::cout << std::left << std::setw(23) << cfg.name << " | "
                  << std::right << std::setw(9) << std::fixed << std::setprecision(1) << acc_pct << "% | "
                  << std::setw(10) << std::setprecision(2) << E << " | "
                  << std::setw(7) << std::setprecision(1) << rej0_pct << "% | "
                  << std::setw(9) << std::setprecision(1) << s0_arg << "% | "
                  << std::setw(10) << std::setprecision(1) << s0_ovl << "% | "
                  << std::setw(9) << std::setprecision(1) << est_tps << " | ";
        for (int s = 0; s < cfg.k_steps; ++s) {
            double sp = st.step_trials[s] > 0 ? 100.0 * (double)st.step_hits[s] / (double)st.step_trials[s] : 0.0;
            std::cout << std::setw(4) << std::setprecision(0) << sp << "% ";
        }
        std::cout << "\n";
    }
    std::cout << "========================================================================================================\n";
    return 0;
}
