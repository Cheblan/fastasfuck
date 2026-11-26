#include "common/check.hpp"
#include "serve/mtp_calib.hpp"
#include "serve/seq_backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

namespace {

using strix::GpuArbiter;
using strix::LogitMasks;
using strix::LogitRows;
using strix::MtpHardwareProfile;
using strix::OnlineMtpLut;
using strix::Top2;

void test_censoring_and_stop_cut() {
    OnlineMtpLut lut;
    const OnlineMtpLut::DraftObs obs[4] = {
        {0, 10},
        {1, 11},
        {2, 12},
        {3, 13},
    };

    const auto pre2 = lut.cell(2, 2, 12);
    const auto pre3 = lut.cell(2, 3, 13);

    // 2 accepted, rejected at step 2 -> step 0,1 hit; step 2 miss; step 3 MUST NOT change.
    lut.update(2, obs, 4, 2, false);
    const auto c1 = lut.cell(2, 1, 11);
    const auto c2 = lut.cell(2, 2, 12);
    const auto c3 = lut.cell(2, 3, 13);

    STRIX_CHECK(c1.trials > OnlineMtpLut::kPriorTrials, "step 1 trial incremented");
    STRIX_CHECK(c2.trials > pre2.trials && c2.hits < pre2.hits, "step 2 miss lowered hits and raised trials");
    STRIX_CHECK(std::abs(c3.hits - pre3.hits) < 1e-6f && std::abs(c3.trials - pre3.trials) < 1e-6f,
                "censored step k=3 must remain untouched");

    // Truncated by stop/length/budget cut after step 0 accepted -> step 1 MUST NOT be marked rejected!
    const auto pre1 = lut.cell(2, 1, 11);
    lut.update(2, obs, 4, 1, true);
    const auto after1 = lut.cell(2, 1, 11);
    STRIX_CHECK(std::abs(after1.hits - pre1.hits) < 1e-6f && std::abs(after1.trials - pre1.trials) < 1e-6f,
                "ended_or_cut must not mark unverified step k=a as rejected");
}

void test_pava_monotonicity() {
    OnlineMtpLut lut;
    for (int i = 0; i < 60; ++i) {
        const OnlineMtpLut::DraftObs low_hit[1] = {{0, 2}};
        lut.update(2, low_hit, 1, 1, false);
        const OnlineMtpLut::DraftObs high_miss[1] = {{0, 12}};
        lut.update(2, high_miss, 1, 0, false);
    }

    for (int r = 0; r < OnlineMtpLut::kRegimes; ++r) {
        for (int k = 0; k < OnlineMtpLut::kMaxK; ++k) {
            float prev = 0.0f;
            for (int b = 0; b < OnlineMtpLut::kBins; ++b) {
                const float p = lut.cond_prob(r, k, b);
                STRIX_CHECK(p >= OnlineMtpLut::kProbMin && p <= OnlineMtpLut::kProbMax, "prob out of bounds");
                STRIX_CHECK(p + 1e-6f >= prev, "PAVA monotonicity violated at r=", r, " k=", k, " b=", b,
                            ": ", p, " < ", prev);
                prev = p;
            }
        }
    }
}

void test_bernoulli_convergence() {
    OnlineMtpLut lut;
    std::mt19937 rng(42);
    std::bernoulli_distribution bern(0.70);

    for (int i = 0; i < 400; ++i) {
        const OnlineMtpLut::DraftObs obs[1] = {{1, 8}};
        const int acc = bern(rng) ? 1 : 0;
        lut.update(2, obs, 1, acc, false);
    }
    const auto c = lut.cell(2, 1, 8);
    const float empirical = c.hits / c.trials;
    STRIX_CHECK(std::abs(empirical - 0.70f) < 0.05f, "Bernoulli p=0.70 converged poorly: ", empirical);
    const float mono = lut.cond_prob(2, 1, 8);
    STRIX_CHECK(mono >= OnlineMtpLut::kProbMin && mono <= OnlineMtpLut::kProbMax, "PAVA probability invalid");
}

void test_regimes_and_bins() {
    STRIX_CHECK(OnlineMtpLut::regime_for_temp(0.0) == 0, "greedy regime");
    STRIX_CHECK(OnlineMtpLut::regime_for_temp(0.3) == 1, "low temp regime");
    STRIX_CHECK(OnlineMtpLut::regime_for_temp(0.6) == 2, "mid temp regime");
    STRIX_CHECK(OnlineMtpLut::regime_for_temp(1.0) == 3, "high temp regime");

    int prev_bin = -1;
    for (float p = 0.0f; p <= 1.0f; p += 0.05f) {
        const int b = OnlineMtpLut::bin_for_prob(p);
        STRIX_CHECK(b >= 0 && b < OnlineMtpLut::kBins, "prob bin out of range");
        STRIX_CHECK(b >= prev_bin, "prob bin mapping must be non-decreasing");
        prev_bin = b;
    }

    prev_bin = -1;
    for (float d = 0.0f; d <= 10.0f; d += 0.5f) {
        const int b = OnlineMtpLut::bin_for_delta_logits(d);
        STRIX_CHECK(b >= 0 && b < OnlineMtpLut::kBins, "bin out of range");
        STRIX_CHECK(b >= prev_bin, "bin mapping must be non-decreasing in delta_logits");
        prev_bin = b;
    }
}

class MockLmBackend final : public strix::LmBackend {
public:
    explicit MockLmBackend(int n) : n_(n) {}
    int64_t capacity() const override { return 1024; }
    int64_t max_chunk() const override { return 512; }
    int64_t logits_row() const override { return 16; }
    std::string describe() const override { return "mock"; }
    int64_t pos() const override { return 0; }
    void reset() override {}
    bool has_mtp() const override { return true; }
    int n_seqs() const override { return n_; }

    std::vector<float> forward_mtp(int32_t token_id, int64_t step) override {
        (void)token_id, (void)step;
        std::vector<float> logits(16, 0.0f);
        logits[1] = 10.0f;
        for (int i = 2; i < 11; ++i) logits[i] = 9.0f;  // 9 flat runners-up!
        return logits;
    }
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override {
        return forward(ids, static_cast<int64_t>(want_logits ? 1 : 0));
    }
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits) override {
        (void)ids;
        return std::vector<float>((size_t)n_logits * 16, 0.0f);
    }
    void save_snapshot(int slot) override { (void)slot; }
    int64_t snapshot_pos(int slot) const override { (void)slot; return -1; }
    bool can_restore_snapshot(int slot) const override { (void)slot; return false; }
    void restore_snapshot(int slot) override { (void)slot; }
    void export_snapshot(int slot, strix::HostBuffer &out, int64_t from) override { (void)slot, (void)out, (void)from; }
    uint64_t snapshot_bytes(int slot, int64_t from) const override { (void)slot, (void)from; return 0; }
    void import_state(const strix::HostBuffer &state, int64_t n, int64_t from) override {
        (void)state, (void)n, (void)from;
    }
    std::string state_fingerprint() const override { return "mock"; }
    Top2 forward_mtp_top2(int seq, int32_t token_id, int64_t step, float inv_T = 1.0f) override {
        (void)seq, (void)token_id, (void)step, (void)inv_T;
        return Top2{1, 5.0f, 1.0f, false, 0.9f};
    }
    std::vector<Top2> forward_mtp_top2_batch(const std::vector<MtpDraftRun> &runs) override {
        if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        return std::vector<Top2>(runs.size(), Top2{1, 5.0f, 1.0f, false, 0.9f});
    }
    std::vector<LogitRows> verify_batch(const std::vector<VerifyRun> &runs, bool cands, int64_t n_valid) override {
        (void)cands, (void)n_valid;
        if (verify_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(verify_sleep_ms));
        std::vector<LogitRows> out;
        out.reserve(runs.size());
        for (const auto &r : runs) {
            out.push_back(LogitRows::from_full(std::vector<float>(r.ids.size() * 16, 0.0f), 16));
        }
        return out;
    }
    void keep_verify(int seq, int64_t rows, int32_t carry) override {
        (void)seq, (void)rows, (void)carry;
    }
    void drop_verify(int seq) override { (void)seq; }

    int sleep_ms = 0;
    int verify_sleep_ms = 0;

private:
    int n_ = 1;
};

void test_gpu_arbiter_zero_wait_and_buckets() {
    // 1. Verify zero-wait in --slots 1 (n == 1)
    {
        MockLmBackend be(1);
        MtpHardwareProfile prof;
        GpuArbiter arb(be);
        arb.set_hw_profile(&prof);
        strix::SeqBackend sb(arb, 0);
        arb.set_decoding(0, true);

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) {
            (void)sb.forward_mtp_top2(1, 0);
            (void)sb.forward_verify_rows({1, 2}, 2, false, 16);
            sb.keep_verify();
            arb.report_slot_acceptance(0, 2);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        STRIX_CHECK(ms < 15.0, "solo n=1 must not hit arbiter coalesce timeout waits (took ", ms, " ms)");

        // k == 0 single-row verify must update bucket 1 round_gpu_ms AND ema_slot_E = 1.0f
        be.verify_sleep_ms = 20;
        for (int i = 0; i < 30; ++i) {
            (void)sb.forward_verify_rows({1}, 1, false, 16);
            sb.keep_verify();
            arb.report_slot_acceptance(0, 1);
        }
        const float tau1 = arb.bucket_tau(1);
        STRIX_CHECK(tau1 > 12.0f && tau1 < 30.0f, "k==0 bucket 1 tau expected near 20 ms, got ", tau1);
        arb.set_decoding(0, false);
    }

    // 2. Verify independent buckets on 4 <-> 1 concurrency switch + outlier clamp
    {
        MockLmBackend be(4);
        MtpHardwareProfile prof;
        GpuArbiter arb(be);
        arb.set_hw_profile(&prof);

        // Warm solo bucket (b = 1) at ~5ms verify, acceptance = 1 -> tau(1) ~ 5ms
        arb.set_decoding(0, true);
        be.verify_sleep_ms = 5;
        for (int i = 0; i < 18; ++i) {
            auto l = arb.verify(0, {0, {1, 2}, nullptr}, false, 16);
            (void)l;
            arb.run(0, GpuArbiter::Kind::Resolve, [&] { be.keep_verify(0, 2, -1); });
            arb.report_slot_acceptance(0, 1);
        }
        const float warm_solo_tau = arb.cluster_tau();
        STRIX_CHECK(warm_solo_tau >= 3.5f && warm_solo_tau <= 8.0f, "solo warm tau expected ~5, got ", warm_solo_tau);

        // Outlier clamp test on solo bucket (rounds >= 16): inject 250ms stall, must clamp at 2x EMA
        be.verify_sleep_ms = 250;
        {
            auto l = arb.verify(0, {0, {1, 2}, nullptr}, false, 16);
            (void)l;
            arb.run(0, GpuArbiter::Kind::Resolve, [&] { be.keep_verify(0, 2, -1); });
            arb.report_slot_acceptance(0, 1);
        }
        const float after_spike = arb.ema_round_gpu_ms();
        STRIX_CHECK(after_spike < 9.0f, "outlier clamp should cap 250ms spike before EMA, got ", after_spike);

        // Switch up to 4 concurrent slots with heavier batch (~40ms, E=1) -> bucket 4 ~40ms, bucket 1 untouched!
        for (int s = 1; s < 4; ++s) arb.set_decoding(s, true);
        be.verify_sleep_ms = 40;
        std::vector<std::thread> ths;
        for (int s = 0; s < 4; ++s) {
            ths.emplace_back([&, s] {
                for (int r = 0; r < 4; ++r) {
                    auto l = arb.verify(s, {s, {1, 2}, nullptr}, false, 16);
                    (void)l;
                    arb.run(s, GpuArbiter::Kind::Resolve, [&] { be.keep_verify(s, 2, -1); });
                    arb.report_slot_acceptance(s, 1);
                }
            });
        }
        for (auto &t : ths) t.join();
        const float batch_ema = arb.ema_round_gpu_ms();
        STRIX_CHECK(batch_ema > 25.0f, "4-slot bucket should reflect ~40ms batch, got ", batch_ema);

        // Drop back to 1 slot -> must IMMEDIATELY return warm solo bucket (~5ms), zero 4->1 drag!
        for (int s = 1; s < 4; ++s) arb.set_decoding(s, false);
        const float restored_solo_tau = arb.cluster_tau();
        STRIX_CHECK(restored_solo_tau < 8.5f, "switching 4->1 must instantly restore bucket 1 tau, got ",
                    restored_solo_tau);
        arb.set_decoding(0, false);
    }

    // 3. Verify flat-tail p_top1(inv_T) sensitivity in LmBackend fallback
    {
        MockLmBackend be(1);
        const Top2 t_hot = be.strix::LmBackend::forward_mtp_top2(0, 0, int64_t{0}, 1.0f);
        const Top2 t_cool = be.strix::LmBackend::forward_mtp_top2(0, 0, int64_t{0}, 1.0f / 0.6f);
        STRIX_CHECK(t_hot.best == 1 && t_cool.best == 1, "best token invariance");
        STRIX_CHECK(t_hot.best_v == 10.0f && t_cool.best_v == 10.0f && t_hot.second_v == 9.0f,
                    "raw logits untouched by inv_T");
        STRIX_CHECK(t_hot.p_top1 > 0.21f && t_hot.p_top1 < 0.25f, "T=1.0 flat tail p_top1 ~0.23, got ", t_hot.p_top1);
        STRIX_CHECK(t_cool.p_top1 > 0.35f && t_cool.p_top1 < 0.39f, "T=0.6 flat tail p_top1 ~0.37, got ", t_cool.p_top1);
    }
}

void test_leviathan_prior_mode() {
    OnlineMtpLut levi(OnlineMtpLut::PriorMode::Leviathan);
    OnlineMtpLut argmax(OnlineMtpLut::PriorMode::Argmax);

    // Flat <think> bins (p_top1 < 0.50, r = 3, k = 0) must start optimistic under Leviathan so the cost gate
    // allows drafting instead of starving at p0 <= 0.375, while Argmax keeps its conservative prior.
    STRIX_CHECK(argmax.cond_prob(3, 0, 0) < 0.10f, "argmax prior bin 0 should stay conservative");
    STRIX_CHECK(levi.cond_prob(3, 0, 0) > 0.55f, "leviathan prior bin 0 expected > 0.55, got ",
                levi.cond_prob(3, 0, 0));
    STRIX_CHECK(levi.cond_prob(3, 0, 7) > 0.70f, "leviathan prior bin 7 expected > 0.70, got ",
                levi.cond_prob(3, 0, 7));

    // Repeated rejects on bin 0 must still calibrate Leviathan downward while maintaining PAVA monotonicity.
    OnlineMtpLut::DraftObs obs[1] = {{0, 0}};
    for (int i = 0; i < 200; ++i) {
        levi.update(3, obs, 1, /*accepted=*/0, /*ended_or_cut=*/false);
    }
    STRIX_CHECK(levi.cond_prob(3, 0, 0) < 0.40f, "leviathan bin 0 failed to adapt down after rejects: ",
                levi.cond_prob(3, 0, 0));
    for (int b = 1; b < OnlineMtpLut::kBins; ++b) {
        STRIX_CHECK(levi.cond_prob(3, 0, b) + 1e-6f >= levi.cond_prob(3, 0, b - 1),
                    "leviathan PAVA monotonicity violated at bin ", b);
    }
}

void test_calib_fingerprint_grouped_threshold() {
    const std::string fp4 = strix::compute_weights_fingerprint(
        "/tmp/nonexistent-weights.faf", "qwen-test", 4, "en+ru", "half-u", nullptr, nullptr, nullptr, 4);
    const std::string fp6 = strix::compute_weights_fingerprint(
        "/tmp/nonexistent-weights.faf", "qwen-test", 4, "en+ru", "half-u", nullptr, nullptr, nullptr, 6);
    STRIX_CHECK(fp4 != fp6, "calibration fingerprint must change when kGroupedMinTokens changes");
}

void test_online_draft_temp_tuner() {
    strix::OnlineDraftTempTuner tuner;
    // Before warmup (kWarmupObs = 28), scale_for must keep the formula fallback (0.70 at T=1.0, 0.88 at T=0.6)
    const float def_t1 = tuner.scale_for(1.0, 0);
    const float def_t06 = tuner.scale_for(0.6, 2);
    STRIX_CHECK(std::abs(def_t1 - 0.70f) < 1e-4f, "T=1.0 fallback expected 0.70, got ", def_t1);
    STRIX_CHECK(std::abs(def_t06 - 0.88f) < 1e-4f, "T=0.6 fallback expected 0.88, got ", def_t06);

    // Feed 27 observations favoring scale index 0 (0.55) on regime=3, step=0 -> still fallback
    const float prefer_0[strix::OnlineDraftTempTuner::kScales] = {0.80f, 0.72f, 0.65f, 0.60f, 0.55f};
    for (int i = 0; i < strix::OnlineDraftTempTuner::kWarmupObs - 1; ++i) {
        tuner.update(3, 0, prefer_0);
        STRIX_CHECK(tuner.active_scale_idx(3, 0) == -1, "tuner switched before warmup ended");
    }
    // 28th observation selects index 0 (0.55), while step=2 stays on fallback 0.70
    tuner.update(3, 0, prefer_0);
    STRIX_CHECK(tuner.active_scale_idx(3, 0) == 0, "warmup should select scale index 0");
    STRIX_CHECK(std::abs(tuner.scale_for(1.0, 0) - 0.55f) < 1e-5f, "step 0 expected 0.55");
    STRIX_CHECK(std::abs(tuner.scale_for(1.0, 2) - 0.70f) < 1e-4f, "step 2 should remain at fallback");

    // Hysteresis test: tiny +0.002 edge at index 3 must NOT flip yet; strong edge (>0.005) eventually flips to 3 (1.00)
    float slight_3[strix::OnlineDraftTempTuner::kScales] = {0.80f, 0.72f, 0.65f, 0.802f, 0.55f};
    tuner.update(3, 0, slight_3);
    STRIX_CHECK(tuner.active_scale_idx(3, 0) == 0, "hysteresis failed: jittered on tiny edge");
    const float strong_3[strix::OnlineDraftTempTuner::kScales] = {0.60f, 0.65f, 0.70f, 0.90f, 0.75f};
    for (int i = 0; i < 60; ++i) tuner.update(3, 0, strong_3);
    STRIX_CHECK(tuner.active_scale_idx(3, 0) == 3, "tuner failed to switch to index 3 after strong shift");
    STRIX_CHECK(std::abs(tuner.scale_for(1.0, 0) - 1.00f) < 1e-5f, "step 0 expected 1.00");

    // Cap check: scale 1.15 (index 4) is allowed at (regime=3, step=0), capped to <= 1.00 (index 3) at mid/step>=1
    const float prefer_4[strix::OnlineDraftTempTuner::kScales] = {0.50f, 0.60f, 0.70f, 0.80f, 0.95f};
    for (int i = 0; i < 120; ++i) {
        tuner.update(3, 0, prefer_4);
        tuner.update(3, 1, prefer_4);
        tuner.update(2, 0, prefer_4);
    }
    STRIX_CHECK(tuner.active_scale_idx(3, 0) == 4 && std::abs(tuner.scale_for(1.0, 0) - 1.15f) < 1e-5f,
                "regime=3 step=0 should permit 1.15");
    STRIX_CHECK(tuner.active_scale_idx(3, 1) == 3 && std::abs(tuner.scale_for(1.0, 1) - 1.00f) < 1e-5f,
                "regime=3 step=1 must cap at 1.00");
    STRIX_CHECK(tuner.active_scale_idx(2, 0) == 3 && std::abs(tuner.scale_for(0.6, 0) - 1.00f) < 1e-5f,
                "regime=2 step=0 must cap at 1.00");
}

}  // namespace

int main() {
    test_censoring_and_stop_cut();
    test_pava_monotonicity();
    test_bernoulli_convergence();
    test_regimes_and_bins();
    test_gpu_arbiter_zero_wait_and_buckets();
    test_leviathan_prior_mode();
    test_calib_fingerprint_grouped_threshold();
    test_online_draft_temp_tuner();
    std::printf("PASS: test_online_mtp_lut\n");
    return 0;
}
