// Leviathan speculative sampling (serve/sampler.hpp sample_draft_cands / verify_leviathan): the output distribution
// must equal the target sampler's exactly, for any draft distribution q, any draft temperature, top_k / top_p.
// Checked statistically: for each scenario, sample_draft_cands + verify_leviathan run N times, and the empirical
// output distribution must match the independently built truncated target distribution p (TVD + chi-square).

#include "common/check.hpp"
#include "serve/sampler.hpp"
#include "serve/seq_backend.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

using strix::LogitRows;
using strix::Sampler;
using strix::SamplingParams;

constexpr int64_t kN = 200000;  // draws per scenario
constexpr int64_t kCands = 20;  // Sampler::kCandidates

// A candidate row (what kernels/logits_topk gives the sampler): ids 1..m ascending, values descending.
LogitRows cand_row(const std::vector<float> &v, const std::vector<int32_t> &id) {
    LogitRows l;
    l.rows = 1, l.row = 248320, l.cands = kCands;
    l.cand_v = v;
    l.cand_id = id;
    l.nan = {0};
    return l;
}

// Independent oracle: the top-k/top-p truncated distribution over a candidate row (the probe's build_top_dist for
// candidates) - deliberately not the sampler's own code, so a bug in trunc_dist_from_cands cannot hide.
struct Dist {
    std::vector<int32_t> id;
    std::vector<double> p;
};

Dist oracle_dist(const std::vector<float> &v, const std::vector<int32_t> &id, double temp, int64_t top_k,
                 double top_p, int64_t n_valid = 248320) {
    std::vector<std::pair<float, int32_t>> valid;
    for (size_t i = 0; i < v.size(); ++i) {
        if (id[i] == INT32_MAX && v[i] == -std::numeric_limits<float>::infinity()) continue;
        if (id[i] < 0 || id[i] >= n_valid || v[i] == -std::numeric_limits<float>::infinity()) continue;
        valid.emplace_back(v[i], id[i]);
    }
    const auto better = [](const std::pair<float, int32_t> &a, const std::pair<float, int32_t> &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    std::sort(valid.begin(), valid.end(), better);
    if (top_k > 0 && valid.size() > (size_t)top_k) valid.resize((size_t)top_k);

    Dist d;
    const double top = (double)valid[0].first / temp;
    std::vector<double> raw(valid.size());
    double sum = 0;
    for (size_t i = 0; i < valid.size(); ++i) {
        raw[i] = std::exp((double)valid[i].first / temp - top);
        sum += raw[i];
    }
    size_t keep = valid.size();
    if (top_p < 1) {
        double acc = 0;
        for (size_t i = 0; i < valid.size(); ++i) {
            acc += raw[i] / sum;
            if (acc >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double kept = 0;
    for (size_t i = 0; i < keep; ++i) kept += raw[i];
    d.id.resize(keep);
    d.p.resize(keep);
    for (size_t i = 0; i < keep; ++i) {
        d.id[i] = valid[i].second;
        d.p[i] = raw[i] / kept;
    }
    return d;
}

// One scenario: target logits (with optional padding tail) + draft logits; runs the pair N times.
struct Scenario {
    const char *name;
    std::vector<float> target_v;   // 20 (or fewer real + padding) candidate logits, descending
    std::vector<int32_t> target_id;
    std::vector<float> draft_v;
    std::vector<int32_t> draft_id;
    double t_target, t_draft;
    int64_t top_k;
    double top_p;
    int64_t n_valid = 248320;
};

void run_scenario(const Scenario &s) {
    LogitRows tgt = cand_row(s.target_v, s.target_id);
    tgt.row = s.n_valid;
    const LogitRows drf = cand_row(s.draft_v, s.draft_id);
    Sampler sampler(SamplingParams{s.t_target, s.top_k, s.top_p, 1234}, s.n_valid);
    std::vector<double> count((size_t)s.n_valid, 0);
    double accepted = 0;
    for (int64_t i = 0; i < kN; ++i) {
        Sampler::TruncDist q;
        const int32_t d = sampler.sample_draft_cands(drf.cand_v.data(), drf.cand_id.data(), drf.cands, false,
                                                      s.t_draft, &q);
        STRIX_CHECK(d >= 0 && d < s.n_valid, s.name, ": draft token ", d, " outside [0, ", s.n_valid, ")");
        const int32_t out = sampler.verify_leviathan(tgt, 0, d, q);
        STRIX_CHECK(out >= 0 && out < s.n_valid, s.name, ": output token ", out, " outside [0, ", s.n_valid, ")");
        accepted += (out == d);
        count[(size_t)out] += 1.0;
    }
    // The truncated target distribution, from the oracle.
    const Dist p = oracle_dist(s.target_v, s.target_id, s.t_target, s.top_k, s.top_p, s.n_valid);
    double tvd = 0, chi2 = 0, support_draws = 0;
    for (size_t i = 0; i < p.id.size(); ++i) {
        const double emp = count[(size_t)p.id[i]] / (double)kN;
        support_draws += count[(size_t)p.id[i]];
        tvd += std::fabs(emp - p.p[i]);
        const double exp_n = p.p[i] * (double)kN;
        const double diff = count[(size_t)p.id[i]] - exp_n;
        chi2 += diff * diff / exp_n;
    }
    STRIX_CHECK((int64_t)support_draws == kN, s.name, ": drew outside target support (", kN - (int64_t)support_draws, ")");
    tvd *= 0.5;
    // E[TVD] ~ sqrt((k-1)/(4N)) per draw set; 0.02 is far above any honest deviation at k <= 20, N = 2e5.
    // chi2: k-1 + 5*sqrt(2(k-1)) is a ~5-sigma bound.
    const double chi2_max = (double)p.id.size() - 1.0 + 5.0 * std::sqrt(2.0 * ((double)p.id.size() - 1.0));
    STRIX_CHECK(tvd < 0.02, s.name, ": TVD to the target distribution ", tvd, ", expected < 0.02");
    STRIX_CHECK(chi2 < chi2_max, s.name, ": chi-square ", chi2, ", expected < ", chi2_max);
    // Acceptance rate ~ sum(min(p, q)) (the Leviathan bound) - loose check (draft support may differ).
    std::printf("%-28s TVD %.5f  chi2 %6.1f (max %.1f)  accept %.3f\n", s.name, tvd, chi2, chi2_max,
                accepted / (double)kN);
}

std::vector<float> ramp(int64_t n, double top_logit, double decay) {
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)(top_logit - decay * (double)i);
    return v;
}

std::vector<int32_t> ids_from(int64_t n, int32_t base) {
    std::vector<int32_t> id((size_t)n);
    for (int64_t i = 0; i < n; ++i) id[(size_t)i] = base + (int32_t)i;
    return id;
}

class MixedDraftMockBackend final : public strix::LmBackend {
public:
    explicit MixedDraftMockBackend(int n) : n_(n) {}
    int64_t capacity() const override { return 1024; }
    int64_t max_chunk() const override { return 512; }
    int64_t logits_row() const override { return 16; }
    std::string describe() const override { return "mixed_mock"; }
    int64_t pos() const override { return 0; }
    void reset() override {}
    bool has_mtp() const override { return true; }
    bool supports_draft_cands() const override { return true; }
    int n_seqs() const override { return n_; }

    std::vector<float> forward(const std::vector<int32_t> &, bool) override { return {}; }
    void save_snapshot(int) override {}
    int64_t snapshot_pos(int) const override { return -1; }
    bool can_restore_snapshot(int) const override { return false; }
    void restore_snapshot(int) override {}
    void export_snapshot(int, strix::HostBuffer &, int64_t) override {}
    uint64_t snapshot_bytes(int, int64_t) const override { return 0; }
    void import_state(const strix::HostBuffer &, int64_t, int64_t) override {}
    std::string state_fingerprint() const override { return "mixed_mock"; }

    std::vector<strix::Top2> forward_mtp_top2_batch(const std::vector<MtpDraftRun> &runs) override {
        std::vector<strix::Top2> out;
        out.reserve(runs.size());
        for (const auto &r : runs) {
            out.push_back(strix::Top2{100 + r.seq, 6.0f, 4.0f, false, 0.85f});
        }
        return out;
    }

    std::vector<MtpDraftCands> forward_mtp_cands_batch(const std::vector<MtpDraftRun> &runs) override {
        ++cands_batches;
        std::vector<MtpDraftCands> out;
        out.reserve(runs.size());
        for (const auto &r : runs) {
            if (r.n_valid > 0) seen_n_valid = r.n_valid;
            MtpDraftCands c{};
            c.best = 200 + r.seq, c.best_v = 8.0f, c.second_v = 7.0f, c.nan = false, c.p_top1 = 0.65f;
            for (int64_t i = 0; i < MtpDraftCands::kCands; ++i) {
                c.id[i] = 200 + r.seq + (int32_t)i;
                c.v[i] = 8.0f - 0.1f * (float)i;
            }
            out.push_back(c);
        }
        return out;
    }

    std::atomic<int64_t> cands_batches{0};
    std::atomic<int64_t> seen_n_valid{0};

private:
    int n_;
};

}  // namespace

int main() {
    // 1. Peaked target, flat draft: most drafts accepted; output = p.
    run_scenario({"peaked target / flat draft", ramp(kCands, 20.0, 1.5), ids_from(kCands, 100),
                  ramp(kCands, 12.0, 0.4), ids_from(kCands, 100), 1.0, 0.7, 20, 0.95});
    // 2. Flat <think>-style fork: three near-equal heads, the case argmax drafting lost ~60% on.
    {
        std::vector<float> t = ramp(kCands, 10.0, 0.0);
        t[0] = 10.0f, t[1] = 9.95f, t[2] = 9.9f, t[3] = 6.0f;
        for (int64_t i = 4; i < kCands; ++i) t[(size_t)i] = (float)(6.0 - 0.5 * (double)(i - 3));
        std::vector<float> q = t;
        q[0] += 0.3f;
        run_scenario({"flat fork (3 heads)", t, ids_from(kCands, 500), q, ids_from(kCands, 500), 1.0, 0.7, 20,
                      0.95});
    }
    // 3. top_p truncates p to a few ids; the draft still proposes all 20.
    {
        std::vector<float> t = ramp(kCands, 12.0, 0.8);  // top_p 0.95 keeps ~6
        run_scenario({"top-p truncates p", t, ids_from(kCands, 7), ramp(kCands, 12.0, 0.6), ids_from(kCands, 7), 0.6,
                      0.42, 20, 0.95});
    }
    // 4. Disjoint supports: every draft rejected, output still = p (residual = p when q misses p's ids).
    run_scenario({"disjoint supports", ramp(kCands, 14.0, 0.9), ids_from(kCands, 1000), ramp(kCands, 14.0, 0.9),
                  ids_from(kCands, 2000), 1.0, 0.7, 20, 0.95});
    // 5. q == p at equal temperature: acceptance 100%, output = d.
    {
        Sampler sampler(SamplingParams{1.0, 20, 0.95, 7}, 248320);
        const LogitRows r = cand_row(ramp(kCands, 11.0, 0.5), ids_from(kCands, 42));
        double acc = 0;
        for (int64_t i = 0; i < 20000; ++i) {
            Sampler::TruncDist q;
            const int32_t d = sampler.sample_draft_cands(r.cand_v.data(), r.cand_id.data(), r.cands, false, 1.0, &q);
            acc += (sampler.verify_leviathan(r, 0, d, q) == d);
        }
        STRIX_CHECK(acc > 19990, "q == p: acceptance ", acc, ", expected ~20000");
        std::printf("%-28s accept %.5f\n", "q == p accepts always", acc / 20000.0);
    }
    // 6. Masked row (padding tail): p over the real candidates only; padding never drawn.
    {
        std::vector<float> t = ramp(5, 13.0, 0.7);
        std::vector<int32_t> tid = ids_from(5, 31);
        for (int64_t i = 5; i < kCands; ++i) t.push_back(-std::numeric_limits<float>::infinity()), tid.push_back(INT32_MAX);
        run_scenario({"masked row (15 padded)", t, tid, ramp(kCands, 13.0, 0.5), ids_from(kCands, 31), 1.0, 0.7, 20,
                      0.95});
    }
    // 7. Same seed, same sequence (RNG determinism of the pair).
    {
        const LogitRows t = cand_row(ramp(kCands, 10.0, 0.4), ids_from(kCands, 3));
        const LogitRows d = cand_row(ramp(kCands, 10.0, 0.9), ids_from(kCands, 3));
        std::vector<int32_t> a, b;
        for (auto *out : {&a, &b}) {
            Sampler s(SamplingParams{1.0, 20, 0.95, 99}, 248320);
            for (int64_t i = 0; i < 500; ++i) {
                Sampler::TruncDist q;
                const int32_t dt = s.sample_draft_cands(d.cand_v.data(), d.cand_id.data(), d.cands, false, 0.7, &q);
                out->push_back(s.verify_leviathan(t, 0, dt, q));
            }
        }
        STRIX_CHECK(a == b, "same seed: the two runs differ");
        std::printf("%-28s ok\n", "seed determinism");
    }
    // 8. Masked row with top_p = 1.0: trailing padding (-inf, INT32_MAX) must not enter TruncDist or emit INT32_MAX.
    {
        std::vector<float> t = ramp(5, 13.0, 0.7);
        std::vector<int32_t> tid = ids_from(5, 31);
        for (int64_t i = 5; i < kCands; ++i) t.push_back(-std::numeric_limits<float>::infinity()), tid.push_back(INT32_MAX);
        run_scenario({"masked row top_p=1.0", t, tid, t, tid, 1.0, 0.7, 20, 1.0});
    }
    // 9. Unsorted / v[0]=-inf target after guard(): must not crash or mis-truncate top_p.
    {
        std::vector<float> t = ramp(kCands, 11.0, 0.4);
        std::vector<int32_t> tid = ids_from(kCands, 70);
        t[0] = -std::numeric_limits<float>::infinity();  // e.g. guard() masked top token
        t[4] = 12.5f;                                    // non-zero slot higher than remaining v[1]
        run_scenario({"guard v[0]=-inf unsorted", t, tid, ramp(kCands, 11.0, 0.4), ids_from(kCands, 70), 1.0, 0.8, 20,
                      0.95});
    }
    // 10. Residual trailing zero probability: q has huge mass on p.id.back() so res.back() == 0.0; verify fallback.
    {
        std::vector<float> t = {10.0f, 9.5f, 9.0f, 8.0f};
        std::vector<int32_t> tid = {10, 11, 12, 13};
        for (int64_t i = 4; i < kCands; ++i) t.push_back(-std::numeric_limits<float>::infinity()), tid.push_back(INT32_MAX);
        std::vector<float> qv = t;
        qv[3] = 16.0f;  // token 13 dominates q -> q(13) >> p(13), so res[3] == 0.0 and rejections happen often
        run_scenario({"residual trailing zero", t, tid, qv, tid, 1.0, 1.0, 4, 1.0});
    }
    // 11. Out-of-vocab draft ids (>= n_valid): unfiltered draft head padding must be stripped before drafting.
    {
        constexpr int64_t kNV = 248077;
        std::vector<float> qv = ramp(kCands, 15.0, 0.3);
        std::vector<int32_t> qid = ids_from(kCands, 100);
        qid[0] = 248100;
        qid[1] = 248200;
        qid[2] = 248319;
        run_scenario({"out-of-vocab draft ids", ramp(kCands, 12.0, 0.4), ids_from(kCands, 100), qv, qid, 1.0, 0.7, 20,
                      0.95, kNV});
    }
    // 12. top_k = 1 at T = 1.0: must deterministically emit target_id[0] for matching and disjoint drafts.
    {
        const LogitRows tgt = cand_row(ramp(kCands, 12.0, 0.5), ids_from(kCands, 600));
        const LogitRows drf_same = cand_row(ramp(kCands, 10.0, 0.5), ids_from(kCands, 600));
        const LogitRows drf_diff = cand_row(ramp(kCands, 10.0, 0.5), ids_from(kCands, 900));
        Sampler sampler(SamplingParams{1.0, 1, 0.95, 55}, 248320);
        for (const auto *drf : {&drf_same, &drf_diff}) {
            for (int i = 0; i < 1000; ++i) {
                Sampler::TruncDist q;
                const int32_t d = sampler.sample_draft_cands(drf->cand_v.data(), drf->cand_id.data(), drf->cands, false,
                                                             0.7, &q);
                const int32_t out = sampler.verify_leviathan(tgt, 0, d, q);
                STRIX_CHECK(out == 600, "top_k=1 expected 600, got ", out);
            }
        }
        std::printf("%-28s ok\n", "top_k=1 T=1.0 deterministic");
    }
    // 13. GpuArbiter mixed concurrency: 2 slots calling draft_cands + 2 calling draft_top2 simultaneously.
    {
        MixedDraftMockBackend be(4);
        strix::GpuArbiter arb(be);
        for (int s = 0; s < 4; ++s) arb.set_decoding(s, true);
        std::vector<std::thread> ths;
        for (int s = 0; s < 4; ++s) {
            ths.emplace_back([&, s] {
                const bool want_c = (s % 2 == 0);
                for (int i = 0; i < 200; ++i) {
                    if (want_c) {
                        const auto c = arb.draft_cands(s, 10 + i, 0, 1.0f, 248077);
                        STRIX_CHECK(c.best == 200 + s && c.id[0] == 200 + s, "mixed cands seq mismatch for slot ", s);
                    } else {
                        const auto t = arb.draft_top2(s, 10 + i, 0, 1.0f);
                        STRIX_CHECK((t.best == 100 + s) || (t.best == 200 + s), "mixed top2 seq mismatch for slot ", s);
                    }
                }
            });
        }
        for (auto &th : ths) th.join();
        for (int s = 0; s < 4; ++s) arb.set_decoding(s, false);
        STRIX_CHECK(be.cands_batches.load() > 0 && be.seen_n_valid.load() == 248077,
                    "mixed arbiter failed to run cands batch or propagate n_valid");
        std::printf("%-28s ok (%lld mixed batches)\n", "GpuArbiter mixed draft",
                    (long long)be.cands_batches.load());
    }
    std::printf("test_leviathan_sampler: all pass\n");
    return 0;
}
