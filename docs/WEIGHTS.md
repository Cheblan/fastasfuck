# FASTASFUCK Model Weights Specification (`.faf`) & Custom GPTQ/GPTAQ Quantization Guide

This document specifies the exact quantization geometry, `.faf` binary layout, calibration pipeline, and **best practices for quantizing custom weights** (fine-tunes, ThinkingCap / Swift / Qwen 3.8-Flash-Next derivatives) for the **fastasfuck** inference engine on AMD Strix Halo silicon (Ryzen AI Max+ 395, `gfx1151`, 128 GB Unified LPDDR5X-8533).

---

## 1. Executive Summary: Hybrid Q6 / Q4 / Q8 + Calibrated FP64 ActOrder GPTQ

- **Weights Container Format:** `.faf` v4 (FastAsFuck unified memory tensor package). Full drop-in compatibility with legacy `.strixw` files and standalone calibrated `weights.faf` replacements (`70,337,409,024` bytes / **`65.51 GiB`**).
- **Target Baseline Model:** Qwen 3.8-Flash-Next (`125B–128.4B` total MoE parameters / `~6B` active per token, 48 hybrid layers: 36 GDN + 12 QSA, 512 routed experts top-10 + 1 shared expert, `mtp.layers.0`).
- **Zero-Overhead Calibration (`static_groups=True`):** Calibrating the three `q4g128` modules (`gdn_in`, `exp_gu`, `exp_down` — **63.54 GiB**, holding >95% of quantization variance) via **FP64 ActOrder GPTQ + Empirical Bayes Shrinkage** cuts held-out KLD vs. BF16 by **`-30.7%`** (`0.031249 → 0.021643`, and `0.021547` in full-FP64 cross-check) with **exact byte parity, zero `g_idx` gather overhead, and identical HIP WMMA kernel throughput**.
- **Prompt Cache Isolation:** Calibrated weights embed `"calibration"` JSON metadata and `"prebuilt_fingerprint"` in the `.faf` v4 header. `Qwen4ExpBackend::state_fingerprint()` (`serve/qwen4exp_backend.cpp`) appends `cal-<prebuilt_fingerprint>` to `.pcache` and `mtp_calib_cache.json` keys so calibrated and uncalibrated weights never collide.

---

## 2. Production Layout Specification (14-Field Schema)

All production Qwen 3.8-Flash-Next builds follow the 14-field layout configuration:

```text
layout = exp_gu=q4g128,exp_down=q4g128,sh_gu=q6g64,sh_down=q6g64,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q6g64,lm_head=q8g64,mtp_fc=q4g64,embed=bf16
```

### Exact Precision, Calibration & Silicon Execution Breakdown

| Layout Field | Sub-Module / Tensor Description | Precision Format | Calibration Method (Prod) | Scale Granularity | Target Silicon Execution |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **`exp_gu`** | Routed MoE Experts: Gate & Up (`512 × [1280, 2560]`) | **`q4g128`** | **FP64 ActOrder GPTQ** (`H_mlp`, `damp=0.05`) | Group 128 (`static_groups`) | Dual SIMD32 `F16` WMMA |
| **`exp_down`** | Routed MoE Experts: Down (`512 × [2560, 640]`) | **`q4g128`** | **ActOrder Shrink512** (`tau=384`, `damp=0.05`) | Group 128 (`static_groups`) | Dual SIMD32 `F16` WMMA |
| **`gdn_in`** | Linear Attention (GDN): Input (`[16480, 2560]`) | **`q4g128`** | **FP64 ActOrder GPTQ** (`H_gdn`, `damp=0.05`) | Group 128 (`static_groups`) | Dual SIMD32 `F16` WMMA |
| **`gdn_out`** | Linear Attention (GDN): Output (`[2560, 6144]`) | **`q6g64`** | Static RTN (~2.5–2.9% RMS error) | Group 64 | WMMA `F16` Accumulator |
| **`sh_gu`** | Shared Expert: Gate & Up (`[1280, 2560]`) | **`q6g64`** | Static RTN | Group 64 | WMMA `F16` Accumulator |
| **`sh_down`** | Shared Expert: Down (`[2560, 640]`) | **`q6g64`** | Static RTN | Group 64 | WMMA `F16` Accumulator |
| **`attn_qkv`** | Full Window Attention: Q, K, V | **`q6g64`** | Static RTN | Group 64 | WMMA `F16` Accumulator |
| **`attn_o`** | Full Window Attention: Output | **`q6g64`** | Static RTN | Group 64 | WMMA `F16` Accumulator |
| **`ple_kv`** | Prompt-Lookup Engine: Key / Value Proj | **`q6g64`** | Static RTN | Group 64 | WMMA `F16` Accumulator |
| **`hc_down`** | Hyper-Connection Mixer: Down | **`q8g64`** | Static RTN (~0.56–0.83% RMS) | Group 64 | Dot2 Vectorized Accumulator |
| **`hc_up`** | Hyper-Connection Mixer: Up | **`q8g64`** (chunk-major) | Static RTN | Group 64 | Dot2 Vectorized Accumulator |
| **`lm_head`** | Language Model Head: Unembedding | **`q8g64`** | Static RTN | Group 64 | High-Precision Dot2 / FP32 |
| **`mtp_fc`** | `mtp.layers.0`: Embedding & Hidden FC | **`q4g64`** | Static RTN (native draft dist.) | Group 64 | Draft WMMA Acceleration |
| **`embed`** | Token Embedding Matrix (`embed_tokens`) | **BF16** | Unquantized | — | Zero Lexical Error |

In addition to the 14 layout fields:
- **MoE Router (`router`):** Native **BF16** (unquantized) to eliminate top-10 gating drift.
- **RMSNorms, Residual Scales, Conv1D Biases, `A_log`, `dt_bias`:** Native **FP32**.
- **N-gram PLE Table (`ngram.table`):** Int8 quantized 8,388,608-row disk/NVMe streaming table (`~48.3 GiB`).

### Quantization Rationale: Why This Split?
1. **`q4g128` Dominates Error & Bandwidth (`63.54 / 65.51 GiB`):** `q4g128` (16 levels) carries `~13.2%–15.9%` relative RMS error, whereas `q6g64` (64 levels) is `~2.5%–2.9%` RMS (`~30×` lower variance) and `q8g64` is `<0.9%` RMS. Calibrating all three `q4g128` classes cuts their Hessian output MSE by **`-49.5%`** (`gdn_in`), **`-48.7%`** (`exp_gu`), and **`-41.5%`** (`exp_down`).
2. **Avoid `Q5` on `gfx1151`:** In `kernels/linear_wmma.hip`, `Q4`, `Q6`, and `Q8` execute the native `F16` WMMA tensor path (`~1,775 tok/s` prefill at 8k), whereas `Q5` falls back to slower `BF16` unpacking (`-5.9%` prefill latency) while increasing memory footprint.
3. **Why Leave `mtp.layers.0` in Native RTN:** Full 48-layer + MTP simulation (`tools/sim_qwen_fp64_mtp.py`) confirms native RTN `mtp.layers.0` already achieves `84.38%` Top-1 agreement against the quantized trunk — within **`0.05 percentage points`** of the unquantized `BF16` MTP ceiling (`84.43%`).

---

## 3. Quantization, Simulation & Verification Toolchain (`tools/`)

| Script / Tool | Role | When to Use |
| :--- | :--- | :--- |
| **[tools/quantize_qwen_gptq.py](../tools/quantize_qwen_gptq.py)** | Production GPU quantizer & bit-exact `Q4_G128` packer (`formats/q4.cpp` parity) | Building calibrated `<faf_tensor>.{q,scale,minv}` shards for any Qwen 3.8 checkpoint (`--mode actorder` or `d02_imp`). |
| **[tools/convert_qwen4exp.cpp](../tools/convert_qwen4exp.cpp)** | `.faf` v4 assembler (`--prebuilt DIR` flag merges calibrated shards with RTN layers) | Packing safetensors + prebuilt GPTQ shards into `weights.faf` and hash-verifying all 853 tensors. |
| **[tools/sim_qwen_quant.py](../tools/sim_qwen_quant.py)** | Exact Qwen4Exp BF16/fake-quant forward engine, `SafetensorsReader`, and held-out `CALIB_DOC` | Core library imported by all simulators & quantizer; baseline schema/PPL sweeps. |
| **[tools/sim_qwen_calib.py](../tools/sim_qwen_calib.py)** | Calibration corpus generator & 50-char sliding-window overlap auditor | Auditing custom calibration corpora for train/eval leakage. |
| **[tools/sim_qwen_awq_gptq.py](../tools/sim_qwen_awq_gptq.py)** | Batched GDN/HC kernels, 61k diverse corpus builder, Round 1/2 AWQ & GPTQ simulator | Prototyping per-tensor MSE/AWQ/GPTQ recipes before writing `.faf`. |
| **[tools/sim_qwen_gptq_r3.py](../tools/sim_qwen_gptq_r3.py)** | CPU FP64 Cholesky/ActOrder (`prepare_actorder_chol_cpu`) + Shrink512 R3 simulator | Reproducing Round 3 ablations (`MoE_Only`, `d02_imp`, `All3_ActOrder_StaticG`). |
| **[tools/sim_qwen_wave2.py](../tools/sim_qwen_wave2.py)** | Fast transposed ActOrder + **GPTAQ** (asymmetric residual compensation) + **CD + Refit** + Q6 hotspot simulator | Testing **GPTAQ**, Coordinate Descent refit, or `q6g64` hotspot calibration on new architectures/corpora before shipping. |
| **[tools/sim_qwen_fp64_mtp.py](../tools/sim_qwen_fp64_mtp.py)** | Strict FP64 Trunk + `mtp.layers.0` calibration & Top-1/KLD draft head evaluator | Verifying speculative draft alignment (`MTP == QuantTrunk`) after changing trunk quantization. |
| **[tools/solve_qwen_quant.py](../tools/solve_qwen_quant.py)** | Multi-objective Pareto solver over bit-width recipes and DRAM footprints | Designing new mixed-precision layouts for different memory caps. |
| **[tools/bench/bench_ab_quant.py](../tools/bench/bench_ab_quant.py)** | Sterile `AGENTS.md`-compliant A/B suite (Bare, Solo MTP, 4-Slot `T=0.6`, 4k/8k Prefill, SHA-256) | Final gate before deploying any newly quantized `weights.faf` to production. |

---

## 4. Best Practices for Quantizing Custom Weights (GPTQ, GPTAQ & Our Scripts)

### Step-by-Step Production Workflow (24 GB GPU, ~35 min end-to-end)

1. **Set ROCm Runtime Guards (Mandatory on RDNA 3 / 7900 XTX & Strix Halo):**
   Peak VRAM is `~5.45 GiB` for `quantize_qwen_gptq.py` (or `~8.5–12.0 GiB` for full BF16+Quant comparative simulators). Always preload `libjemalloc.so` and disable ROCprofiler HSA intercepts:
   ```bash
   export LD_PRELOAD=/usr/lib/libjemalloc.so
   export ROCP_TOOLS_LIB=""
   export HSA_TOOLS_LIB=""
   export ROCP_HSA_INTERCEPT=0
   ```

2. **Generate Calibrated `Q4_G128` Components:**
   - Use **`--mode actorder`** (default, lowest held-out KLD `-30.7%`, highest Top-1 `95.85%`) for general coding, agentic, and reasoning checkpoints, or **`--mode d02_imp`** (`-23.5%` KLD, `+0.000` dPPL) if strict raw perplexity parity matters more than logit alignment.
   ```bash
   python3 -u tools/quantize_qwen_gptq.py \
     --src /path/to/Custom-Qwen3.8-BF16 \
     --out /mnt/nvme2/tmp-gptq-q4 \
     --mode actorder
   ```
   *(On a 24 GB GPU, `--expert-chunk 64` and `40 × 1536 = 61,440` calibration tokens completes all 48 layers in ~30–35 minutes and produces `calib_meta.json`, `calib_manifest.json`, and `layer_stats.json` alongside the prebuilt `.safetensors` shards.)*

3. **Assemble & Hash-Verify `.faf` v4:**
   ```bash
   ./build/strix/convert_qwen4exp \
     --src /path/to/Custom-Qwen3.8-BF16 \
     --out /path/to/Custom-Qwen3.8-faf \
     --prebuilt /mnt/nvme2/tmp-gptq-q4 \
     --git "$(git rev-parse --short HEAD)" \
     --layout exp_gu=q4g128,exp_down=q4g128,sh_gu=q6g64,sh_down=q6g64,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q6g64,lm_head=q8g64,mtp_fc=q4g64,embed=bf16
   ```
   Verify `prebuilt_fingerprint` and `"calibration"` metadata via:
   ```bash
   ./build/strix/inspect_faf /path/to/Custom-Qwen3.8-faf/weights.faf | head -n 20
   ```

4. **Run Sterile A/B Verification (`AGENTS.md` Standard):**
   Start `fastasfuck_server` with **`--prompt-cache off`** (never `--prompt-cache-gib 0`, which can purge cached `.pcache` files on disk!) and run:
   ```bash
   python3 -u tools/bench/bench_ab_quant.py \
     --url http://127.0.0.1:5300 \
     --model <model-id> \
     --suite all \
     --output ab_verify.json
   ```
   Confirm `nonzero_cache_count == 0`, `all_deterministic == true` (`5/5` SHA-256 bit-identity at `T = 0.0`), parity on `verify_base_ms` / Solo Bare (`~39.9 tok/s`) / 4k & 8k Prefill, and healthy 4-slot concurrent MTP stability.

---

### GPTQ vs. GPTAQ vs. CD+Refit vs. AWQ — Empirical Rules & Pitfalls

Every method below was implemented and evaluated identically across all 48 layers of `Thinking-Cap-Qwen-3.8-FN` (`T_cal = 61,440`, held-out `CALIB_DOC` `T_eval = 1,807`, unquantized `BF16 PPL = 3.1131`):

| Method / Recipe | Mean KLD vs BF16 | Δ vs RTN | Med KLD | p95 KLD | PPL | Top-1 vs BF16 | Engineering Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BF16 Reference` | `0.000000` | — | `0.000000` | `0.000000` | `3.1131` | `100.00%` | Ground-truth ceiling |
| `0_RTN_Prod` (uncalibrated) | `0.031249` | `0.0%` | `0.001346` | `0.124079` | `3.1163` | `94.80%` | Baseline |
| `AWQ / MSEClip` (61k, unweighted) | `0.041136` | `+31.6%` | `0.001619` | `0.134862` | `3.3850` | `94.50%` | **Reject** — group clipping hurts static `q4g128` |
| `GPTAQ` (`sim_qwen_wave2.py`, `alpha=0.25, damp=0.10`) | `0.029896` | `-4.3%` | `0.001367` | `0.098775` | `3.2413` | `95.57%` | **Overfits 512-expert MoE** at 61k tokens |
| `CD + BF16 Refit` (`sim_qwen_wave2.py`) | `0.028286` | `-9.5%` | `0.001262` | `0.107421` | `3.0151` | `94.63%` | **Overfits** — cuts cal MSE `-5.5%` but degrades eval KLD |
| `FP32-Hessian ActOrder` (bug check) | `0.025477` | `-18.5%` | `0.001200` | `0.101840` | `3.0892` | `95.02%` | **Reject FP32** — loses Cholesky precision at `H~71k` |
| `FP64 ActOrder + Q6_Hotspots` (`gdn_out/attn_o/sh_d`) | `0.022498` | `-28.0%` | `0.001116` | `0.074003` | `3.1342` | `96.01%` | Tradeoff — tighter p95, slightly worse Mean KLD |
| **`FP64 ActOrder StaticG + Shrink512` (`--mode actorder`)** | **`0.021643`** | **`-30.7%`** | **`0.001175`** | **`0.084673`** | `3.1746` | **`95.85%`** | **Production Winner** (`0.021547` / `PPL 3.1093` in full-FP64) |

#### Calibrated Release Matrix (All 3 Qwen 3.8-Flash-Next Models, `CALIB_DOC`, `T_eval = 1,807`)

| Model | Variant | Mean KLD vs BF16 | Δ vs RTN | p95 KLD | Max KLD | PPL (Δ vs BF16) | Top-1 | Top-5 | MTP `q8` Top-1 / KLD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ThinkingCap-Qwen3.8-FN** | `RTN` → **`GPTQ`** | `0.031249 →` **`0.021643`** | **`-30.7%`** | `0.084673` | `2.674 →` **`1.987`** | `3.1163 → 3.1746` *(full-FP64 `3.1093`)* | **`95.85%`** | **`99.94%`** | `83.32%` / **`0.4076`** |
| **Swift-1.5-Qwen3.8-FN** | `RTN` → **`GPTQ`** | `0.037388 →` **`0.021508`** | **`-42.5%`** | `0.1287 →` **`0.0864`** | `4.640 →` **`1.661`** | `2.9925 →` **`2.9986`** *(BF16 `3.0020`)* | `94.74%` | **`99.94%`** | **`85.43%`** / **`0.3787`** |
| **Qwen3.8-Flash-Next (Base)** | `RTN` → **`GPTQ`** | `0.030376 →` **`0.021393`** | **`-29.6%`** | `0.1113 →` **`0.0849`** | `3.596 →` **`1.539`** | `3.0327 → 3.0508` *(BF16 `3.0119`)* | **`95.40%`** | **`99.94%`** | `84.82%` / `0.4017` |

#### Critical Rules When Quantizing Your Own Weights or Tuning GPTAQ

1. **Always Accumulate Hessians Strictly in FP64 (`.double()`):**
   Activation vectors `u` in Qwen 3.8 contain channel outliers reaching RMS `~266`, giving diagonal Hessian entries $X^T X \approx 71{,}000$ alongside small tail eigenvalues (`~0.05`). Accumulating `u.float().T @ u.float()` in FP32 drops mantissa precision on tail curvature, warps Cholesky inverse columns, and shifts ActOrder rankings — costing `+0.0039` Mean KLD (`0.02155 → 0.02548`). Always accumulate `H_gdn`, `H_mlp`, and dense Hessians in CPU/GPU `float64`.

2. **How and When to Use GPTAQ (`tools/sim_qwen_wave2.py`):**
   - **What GPTAQ does:** Instead of minimizing $\|(Q - W) X\|_F^2$ with symmetric Hessian $H = X^T X$, GPTAQ aligns the quantized layer trajectory back toward unquantized BF16 activations $\hat{Y} \approx W_{\text{bf16}} X_{\text{bf16}}$ using an asymmetric cross-covariance correction driven by `alpha_aq` (`H_asym = X_q^T X_q` paired with residual $dY$ compensation weighted by $\alpha$).
   - **Why GPTAQ lost to symmetric FP64 ActOrder on Qwen 3.8 (`0.0299` vs `0.0216`):** With 512 fine-grained experts and `61,440` tokens, each routed expert averages ~1,200 tokens (`< D`), so compensating cumulative cross-layer trajectory drift amplifies calibration-sample variance inside `exp_down` and blows up out-of-domain KLD even while lowering in-sample Hessian MSE.
   - **Actionable Rule:**
     - For **512-expert MoE / fine-tunes** at 32k–64k calibration tokens: use **symmetric FP64 ActOrder (`static_groups=True`) + Shrink512** (`quantize_qwen_gptq.py --mode actorder`). Never enable GPTAQ or CD+Refit blindly.
     - Only experiment with **GPTAQ** (`tools/sim_qwen_wave2.py`) if calibrating a **dense model** (or narrow `≤16` expert MoE) with **`≥256k` domain-matched tokens**, higher damping (`damp_aq >= 0.10`), and conservative mixing (`alpha_aq <= 0.25`) — and **always gate by held-out KLD on an unseen document, never calibration MSE**.

3. **Keep Cholesky/Solve on CPU FP64 (`prepare_actorder_chol_cpu`):**
   ROCm `torch.linalg.solve` / `TRSM` in FP64 triggers `Memory access fault` intermittently under batched expert workloads. Our scripts accumulate `H` efficiently via matmul, run symmetric Cholesky/inverse cleanly on CPU `float64`, and run the transposed contiguous ActOrder error propagation on GPU in `float32`.

4. **Stream `experts.gate_up_proj` & `experts.down_proj` in Chunks (`read_slice_dim0`):**
   Unpacked 512-expert FP32/BF16 weights exceed `2 GiB` per tensor (`512 × 1280 × 2560 × 2B = 3.2 GiB`). Never call full `read_tensor()` (which truncates at 2 GiB on large shards); always stream in 64-expert chunks via `SafetensorsReader.read_slice_dim0()`.

5. **Corpus Construction Rules:**
   - Cover every target workload present in production (coding, multi-turn agentic/tool-use traces, math, long-form docs, multilingual such as Russian).
   - Ensure `T_cal >= 61,440` (`40 × 1536`) so `H_mlp` (`2560 × 2560`) and `H_gdn` (`2560 × 2560`) have `>24×` oversampling.
   - Run the disjointness check (`tools/sim_qwen_calib.py`) so zero 50-char n-grams leak from held-out eval into `H`.

6. **Always Keep `static_groups=True`:**
   Dynamic permuted group grids (`actorder` with `g_idx` row reordering) break contiguous `Q4_G128` SIMD loading and destroy Strix Halo memory bandwidth. `static_groups=True` freezes `[0..127], [128..255], ...` BF16 `(scale, minv)` grids first and compensates quantization error across columns in importance order — retaining 100% of kernel speed.

---

## 5. Serving Configuration

In `deploy/fastasfuck-server.conf`:

```ini
# Path to packed .faf weights (RTN or calibrated GPTQ drop-in)
weights = /models/Thinking-Cap-Qwen-3.8-FN-faf/weights.faf

# Dedicated n-gram streaming table on NVMe
ngram = /models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table

# Tokenizer and generation metadata
tokenizer = /models/Thinking-Cap-Qwen-3.8-FN-faf/tokenizer.json
generation-config = /models/Thinking-Cap-Qwen-3.8-FN-faf/generation_config.json
```
