#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "serve/json.hpp"

namespace strix {

struct MtpCalibrationConfig {
    bool enabled = true;
    int draft_samples = 300;
    int verify_samples_min = 25;
    int verify_samples_max = 80;
    int warmup_samples = 15;
    float tolerance_ms = 0.08f;
    int triplets_target = 1000;
    bool cache_enabled = true;
    std::string cache_path;

    float fallback_draft_ms = 1.64f;
    float fallback_verify_base_ms = 26.84f;
    float fallback_verify_step_ms = 4.30f;
    std::vector<float> fallback_verify_grid;
    float fallback_alpha = 0.85f;
    float fallback_beta = -0.04f;
    float fallback_bias = 0.0f;
};

struct MtpHardwareProfile {
    float draft_step_ms = 1.64f;
    float verify_base_ms = 26.84f;
    float verify_step_ms = 4.30f;
    std::vector<float> verify_latencies = {
        0.0f, 26.84f, 32.13f, 36.62f, 41.60f, 46.11f, 46.89f, 50.04f, 54.33f
    };

    float get_verify_cost(int64_t T) const {
        if (T <= 0) return 0.0f;
        if ((size_t)T < verify_latencies.size()) {
            return verify_latencies[(size_t)T];
        }
        return verify_latencies.back() + (float)(T - (int64_t)(verify_latencies.size() - 1)) * verify_step_ms;
    }
};

struct MtpAcceptanceProfile {
    float alpha = 0.85f;
    float beta = -0.04f;
    float bias = 0.0f;
    int sample_count = 0;
};

struct MtpCalibratedProfile {
    std::string fingerprint;
    std::string model_id;
    std::string weights_path;
    uint64_t weights_strix_hash = 0;
    uint64_t weights_size = 0;
    int64_t weights_mtime = 0;
    int draft_head_bits = 4;
    bool draft_head_q4 = true;
    std::string mtp_vocab_filter = "all";
    std::string gdn_precision = "half-u";

    MtpHardwareProfile hw;
    MtpAcceptanceProfile acceptance;
};

float compute_median(std::vector<float> values);
float compute_trimmed_mean(std::vector<float> values, float trim_pct = 0.10f);
std::vector<float> filter_iqr_outliers(const std::vector<float>& values, float k = 1.5f);

std::string compute_weights_fingerprint(
    const std::string& weights_path,
    const std::string& model_id,
    int draft_head_bits,
    const std::string& mtp_vocab_filter,
    const std::string& gdn_precision,
    uint64_t* out_strix_hash = nullptr,
    uint64_t* out_file_size = nullptr,
    int64_t* out_mtime = nullptr
);

struct TripletSample {
    int64_t step;
    float delta_logits;
    int accepted;
};

MtpAcceptanceProfile fit_acceptance_model(
    const std::vector<TripletSample>& samples,
    float l2_reg = 1e-4f
);

std::optional<MtpCalibratedProfile> load_cached_profile(
    const std::string& cache_file,
    const std::string& fingerprint
);

bool save_cached_profile(
    const std::string& cache_file,
    const MtpCalibratedProfile& profile
);

} // namespace strix
