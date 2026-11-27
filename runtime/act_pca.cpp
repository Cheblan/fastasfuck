#include "runtime/act_pca.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

namespace strix {

namespace {

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// Classic EISPACK Householder tridiagonalization (0-based n x n symmetric matrix z, outputs d[n], e[n], eigenvectors z).
void tred2(int n, std::vector<double> &z, std::vector<double> &d, std::vector<double> &e) {
    auto at = [&](int r, int c) -> double & { return z[(size_t)r * (size_t)n + (size_t)c]; };
    for (int i = n - 1; i > 0; --i) {
        const int l = i - 1;
        double h = 0.0;
        double scale = 0.0;
        if (l > 0) {
            for (int k = 0; k <= l; ++k) scale += std::abs(at(i, k));
            if (scale == 0.0) {
                e[(size_t)i] = at(i, l);
            } else {
                for (int k = 0; k <= l; ++k) {
                    at(i, k) /= scale;
                    h += at(i, k) * at(i, k);
                }
                const double f = at(i, l);
                double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
                e[(size_t)i] = scale * g;
                h -= f * g;
                at(i, l) = f - g;
                double ff = 0.0;
                for (int j = 0; j <= l; ++j) {
                    at(j, i) = at(i, j) / h;
                    double g2 = 0.0;
                    for (int k = 0; k <= j; ++k) g2 += at(j, k) * at(i, k);
                    for (int k = j + 1; k <= l; ++k) g2 += at(k, j) * at(i, k);
                    e[(size_t)j] = g2 / h;
                    ff += e[(size_t)j] * at(i, j);
                }
                const double hh = ff / (h + h);
                for (int j = 0; j <= l; ++j) {
                    const double f2 = at(i, j);
                    const double g2 = e[(size_t)j] - hh * f2;
                    e[(size_t)j] = g2;
                    for (int k = 0; k <= j; ++k) {
                        at(j, k) -= (f2 * e[(size_t)k] + g2 * at(i, k));
                    }
                }
            }
        } else {
            e[(size_t)i] = at(i, l);
        }
        d[(size_t)i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        if (d[(size_t)i] != 0.0) {
            for (int j = 0; j < i; ++j) {
                double g = 0.0;
                for (int k = 0; k < i; ++k) g += at(i, k) * at(k, j);
                for (int k = 0; k < i; ++k) at(k, j) -= g * at(k, i);
            }
        }
        d[(size_t)i] = at(i, i);
        at(i, i) = 1.0;
        for (int j = 0; j < i; ++j) {
            at(j, i) = 0.0;
            at(i, j) = 0.0;
        }
    }
}

// Implicit QL eigensolver on tridiagonal (d, e) from tred2; accumulates orthogonal eigenvectors in column-indexed z.
void tql2(int n, std::vector<double> &d, std::vector<double> &e, std::vector<double> &z) {
    auto at = [&](int r, int c) -> double & { return z[(size_t)r * (size_t)n + (size_t)c]; };
    for (int i = 1; i < n; ++i) e[(size_t)(i - 1)] = e[(size_t)i];
    e[(size_t)(n - 1)] = 0.0;
    for (int l = 0; l < n; ++l) {
        int iter = 0;
        for (;;) {
            int m = l;
            for (; m < n - 1; ++m) {
                const double dd = std::abs(d[(size_t)m]) + std::abs(d[(size_t)(m + 1)]);
                if (std::abs(e[(size_t)m]) + dd == dd) break;
            }
            if (m == l) break;
            STRIX_CHECK(++iter <= 64, "tql2: eigenvalue failed to converge after 64 iterations at row ", l);
            double g = (d[(size_t)(l + 1)] - d[(size_t)l]) / (2.0 * e[(size_t)l]);
            double r = std::hypot(g, 1.0);
            g = d[(size_t)m] - d[(size_t)l] + e[(size_t)l] / (g + std::copysign(r, g));
            double s = 1.0, c = 1.0, p = 0.0;
            int i = m - 1;
            for (; i >= l; --i) {
                double f = s * e[(size_t)i];
                double b = c * e[(size_t)i];
                r = std::hypot(f, g);
                e[(size_t)(i + 1)] = r;
                if (r == 0.0) {
                    d[(size_t)(i + 1)] -= p;
                    e[(size_t)m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[(size_t)(i + 1)] - p;
                r = (d[(size_t)i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[(size_t)(i + 1)] = g + p;
                g = c * r - b;
                for (int k = 0; k < n; ++k) {
                    const double t = at(k, i + 1);
                    at(k, i + 1) = s * at(k, i) + c * t;
                    at(k, i) = c * at(k, i) - s * t;
                }
            }
            if (r == 0.0 && i >= l) continue;
            d[(size_t)l] -= p;
            e[(size_t)l] = g;
            e[(size_t)m] = 0.0;
        }
    }
}

struct CacheHeader {
    char magic[4];
    uint32_t version;
    int64_t R;
    int64_t K;
    uint32_t fp_len;
    uint32_t reserved;
};

}  // namespace

std::vector<uint16_t> compute_act_pca_basis_bf16(const uint16_t *u_bf16, int64_t N, int64_t K, int64_t R) {
    STRIX_CHECK(u_bf16 != nullptr && N >= 1 && K >= 1 && R >= 1 && R <= K,
                "compute_act_pca_basis_bf16: invalid args N=", N, ", K=", K, ", R=", R);

    constexpr int64_t kMaxSamples = 1024;
    const int64_t n_use = std::min(N, kMaxSamples);
    std::vector<double> U((size_t)(n_use * K));
    for (int64_t i = 0; i < n_use; ++i) {
        const int64_t src_i = (n_use == N) ? i : (i * N) / n_use;
        const uint16_t *sp = u_bf16 + src_i * K;
        double *dp = U.data() + i * K;
        for (int64_t k = 0; k < K; ++k) dp[k] = (double)bf16_to_f32(sp[k]);
    }

    // Dual Gram matrix G = U * U^T [n_use, n_use] in FP64
    const int n = (int)n_use;
    std::vector<double> G((size_t)n * (size_t)n, 0.0);
    const unsigned hw = std::max(1u, std::min<unsigned>(16u, std::thread::hardware_concurrency()));
    {
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < hw; ++t) {
            workers.emplace_back([&, t]() {
                for (int i = (int)t; i < n; i += (int)hw) {
                    const double *ui = U.data() + (size_t)i * (size_t)K;
                    for (int j = 0; j <= i; ++j) {
                        const double *uj = U.data() + (size_t)j * (size_t)K;
                        double s = 0.0;
                        for (int64_t k = 0; k < K; ++k) s += ui[k] * uj[k];
                        G[(size_t)i * (size_t)n + (size_t)j] = s;
                        G[(size_t)j * (size_t)n + (size_t)i] = s;
                    }
                }
            });
        }
        for (auto &w : workers) w.join();
    }

    std::vector<double> evals((size_t)n, 0.0), e_off((size_t)n, 0.0);
    tred2(n, G, evals, e_off);
    tql2(n, evals, e_off, G);

    std::vector<int> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return evals[(size_t)a] > evals[(size_t)b]; });

    const double max_ev = std::max(0.0, evals[(size_t)order[0]]);
    const double ev_floor = std::max(1e-24, max_ev * 1e-12);
    int n_pos = 0;
    while (n_pos < n && n_pos < (int)R && evals[(size_t)order[(size_t)n_pos]] > ev_floor) ++n_pos;

    // Recover right singular vectors v_r = U^T * eigvec_r in FP64 (parallel)
    std::vector<double> E_f64((size_t)(R * K), 0.0);
    {
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < hw; ++t) {
            workers.emplace_back([&, t]() {
                for (int r = (int)t; r < n_pos; r += (int)hw) {
                    const int col = order[(size_t)r];
                    double *vp = E_f64.data() + (size_t)r * (size_t)K;
                    for (int i = 0; i < n; ++i) {
                        const double w = G[(size_t)i * (size_t)n + (size_t)col];
                        const double *ui = U.data() + (size_t)i * (size_t)K;
                        for (int64_t k = 0; k < K; ++k) vp[k] += w * ui[k];
                    }
                }
            });
        }
        for (auto &w : workers) w.join();
    }

    // Normalize recovered eigenvectors + Modified Gram-Schmidt re-orthogonalization + rank-deficiency padding
    int valid_rows = 0;
    for (int r = 0; r < n_pos; ++r) {
        double *vp = E_f64.data() + (size_t)valid_rows * (size_t)K;
        if (valid_rows != r) std::memcpy(vp, E_f64.data() + (size_t)r * (size_t)K, (size_t)K * sizeof(double));
        for (int p = 0; p < valid_rows; ++p) {
            const double *prev = E_f64.data() + (size_t)p * (size_t)K;
            double dot = 0.0;
            for (int64_t k = 0; k < K; ++k) dot += vp[k] * prev[k];
            for (int64_t k = 0; k < K; ++k) vp[k] -= dot * prev[k];
        }
        double n2 = 0.0;
        for (int64_t k = 0; k < K; ++k) n2 += vp[k] * vp[k];
        if (n2 > 1e-18) {
            const double inv = 1.0 / std::sqrt(n2);
            for (int64_t k = 0; k < K; ++k) vp[k] *= inv;
            ++valid_rows;
        }
    }
    int64_t pad_k = 0;
    while (valid_rows < (int)R && pad_k < K) {
        double *vp = E_f64.data() + (size_t)valid_rows * (size_t)K;
        std::fill(vp, vp + K, 0.0);
        vp[pad_k++] = 1.0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int p = 0; p < valid_rows; ++p) {
                const double *prev = E_f64.data() + (size_t)p * (size_t)K;
                double dot = 0.0;
                for (int64_t k = 0; k < K; ++k) dot += vp[k] * prev[k];
                for (int64_t k = 0; k < K; ++k) vp[k] -= dot * prev[k];
            }
        }
        double n2 = 0.0;
        for (int64_t k = 0; k < K; ++k) n2 += vp[k] * vp[k];
        if (n2 > 1e-12) {
            const double inv = 1.0 / std::sqrt(n2);
            for (int64_t k = 0; k < K; ++k) vp[k] *= inv;
            ++valid_rows;
        }
    }
    STRIX_CHECK(valid_rows == R, "compute_act_pca_basis_bf16: failed to complete orthonormal basis (", valid_rows,
                " < ", R, ")");

    std::vector<uint16_t> E_bf16((size_t)(R * K));
    for (size_t i = 0; i < E_bf16.size(); ++i) E_bf16[i] = f32_to_bf16((float)E_f64[i]);
    return E_bf16;
}

bool load_act_pca_cache(const std::string &path, const std::string &fingerprint, int64_t R, int64_t K,
                        std::vector<uint16_t> &out_bf16) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    CacheHeader h{};
    if (std::fread(&h, sizeof(h), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }
    if (std::memcmp(h.magic, "SPCA", 4) != 0 || h.version != 2 || h.R != R || h.K != K ||
        h.fp_len != (uint32_t)fingerprint.size()) {
        std::fclose(fp);
        return false;
    }
    std::string fbuf(h.fp_len, '\0');
    if (h.fp_len > 0 && std::fread(fbuf.data(), 1, h.fp_len, fp) != h.fp_len) {
        std::fclose(fp);
        return false;
    }
    if (fbuf != fingerprint) {
        std::fclose(fp);
        return false;
    }
    std::vector<uint16_t> tmp((size_t)(R * K));
    if (std::fread(tmp.data(), sizeof(uint16_t), tmp.size(), fp) != tmp.size()) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    out_bf16 = std::move(tmp);
    return true;
}

void save_act_pca_cache(const std::string &path, const std::string &fingerprint, int64_t R, int64_t K,
                        const std::vector<uint16_t> &e_bf16) {
    STRIX_CHECK((int64_t)e_bf16.size() == R * K, "save_act_pca_cache: size mismatch");
    const std::string tmp_path = path + ".tmp";
    FILE *fp = std::fopen(tmp_path.c_str(), "wb");
    if (!fp) return;
    CacheHeader h{};
    std::memcpy(h.magic, "SPCA", 4);
    h.version = 2;
    h.R = R;
    h.K = K;
    h.fp_len = (uint32_t)fingerprint.size();
    bool ok = (std::fwrite(&h, sizeof(h), 1, fp) == 1);
    if (ok && h.fp_len > 0) ok = (std::fwrite(fingerprint.data(), 1, h.fp_len, fp) == h.fp_len);
    if (ok) ok = (std::fwrite(e_bf16.data(), sizeof(uint16_t), e_bf16.size(), fp) == e_bf16.size());
    std::fclose(fp);
    if (ok) std::rename(tmp_path.c_str(), path.c_str());
    else std::remove(tmp_path.c_str());
}

std::string default_act_pca_calibration_text() {
    return
        "<|im_start|>user\n"
        "Write an asynchronous Python 3.12 task DAG using asyncio with cycle detection and semaphore bounds.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "We sort nodes topologically via Kahn's algorithm, guard concurrency with asyncio.Semaphore, "
        "and cancel downstream dependents on exception while independent branches finish normally.\n"
        "</think>\n"
        "```python\n"
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class DagNode:\n"
        "    name: str\n"
        "    deps: tuple[str, ...] = ()\n"
        "    async def execute(self, res: dict[str, object]) -> object:\n"
        "        await asyncio.sleep(0.01)\n"
        "        return {d: res[d] for d in self.deps}\n"
        "```\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Prove the Riemann-Lebesgue Lemma for f in L1(R) and outline x86_64 4-level page table translation.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "For \\(\\hat{f}(\\xi)=\\int f(x)e^{-i\\xi x}dx\\), step functions \\(g=\\sum c_k\\mathbf{1}_{[a_k,b_k]}\\) are dense in L1(R): "
        "\\(|\\hat{f}(\\xi)|\\le\\|f-g\\|_1+\\sum|c_k|\\cdot 2/|\\xi|\\to 0\\). On x86_64, 48-bit VAs index PGD[47:39], PUD[38:30], PMD[29:21], PTE[20:12], "
        "offset[11:0] starting at CR3; TLB shootdowns use SMP IPIs and faults enter do_page_fault -> handle_mm_fault (COW/anon).\n"
        "</think>\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Переведи на научный русский язык и верни JSON-вызов для метеоархива и мультиязычной сводки.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "В распределённых базах данных и астрофизическом моделировании проекция активаций в ортонормированный базис сохраняет точное ранжирование топ-кандидатов. Подготовим вызов инструмента.\n"
        "</think>\n"
        "<tool_call>\n"
        "{\"name\": \"fetch_weather_bulletin\", \"arguments\": {\"station\": \"NO-OSL-42\", \"iso_date\": \"2025-09-14\", \"fields\": [\"pressure_hpa\", \"wind_kt\"], \"strict\": true}}\n"
        "</tool_call>\n"
        "{\"status\": 200, \"ledger_id\": \"acc_78412\", \"checksum\": \"a94a8fe5ccb19ba61c4c\", \"token_b64\": \"QWxhZGRpbjpvcGVuIHNlc2FtZQ==\"}\n"
        "天文学家利用高精度光谱仪观测遥远星系的红移效应，并通过分布式关系型数据库的B+树索引加速海量星表检索与气象雷达回波分析。"
        "編譯器在進行暫存器配置時會建構干擾圖並透過著色演算法減少記憶體溢出。"
        "人工衛星の軌道計算では大気抵抗と重力摂動を数値積分で解き、コンパイラの静的単一代入形式と組み合わせて高速化する。"
        "기상 예보 시스템은 수치 모델과 다중 버전 동시성 제어 데이터베이스를 결합하여 관측 자료를 실시간으로 처리한다.\n"
        "في قواعد البيانات العلائقية، يضمن فهرس الشجرة المتوازنة سرعة الاستعلام وتكامل المعاملات. "
        "Στη θεωρία πληροφορίας ισχύει ∀x∈ℝⁿ, ∃λ>0 : ∥Ax−b∥₂ ≤ ∂f/∂t ⊕ ∑ᵢ αᵢ⊗βᵢ. "
        "במערכות קבצים מבוזרות, שכפול הנתונים שומר על עקביות. "
        "खगोलीय वेधशालाओं में तारों के स्पेक्ट्रम का सटीक विश्लेषण किया जाता है। "
        "การพยากรณ์อากาศอาศัยข้อมูลดาวเทียมและการคำนวณเชิงตัวเลขอย่างต่อเนื่อง\n"
        "┌──────┬────────┐\n"
        "│ Node │ Status │\n"
        "├──────┼────────┤\n"
        "│ DB01 │ OK ✓ ☀ │\n"
        "└──────┴────────┘ α≠β ⇒ ∞ ⊥ ∈ ℂ\n"
        "<|im_end|>\n";
}

}  // namespace strix
