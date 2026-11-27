# fastasfuck

<p align="center">
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-6f42c1"></a>
  <img alt="Platform: Arch Linux, AMD Strix Halo (gfx1151)" src="https://img.shields.io/badge/platform-Arch%20Linux%20%C2%B7%20Strix%20Halo%20gfx1151-1abc9c">
  <a href="https://huggingface.co/collections/Cheblan/fastasfuck-quantizations"><img alt="Hugging Face Collection" src="https://img.shields.io/badge/weights-Hugging%20Face%20Collection-ffcc4d"></a>
  <a href="https://github.com/Cheblan/fastasfuck"><img alt="Repository" src="https://img.shields.io/badge/github-fastasfuck-blue"></a>
</p>

<h2 align="center">High-Throughput Multi-Agent LLM Engine for AMD Strix Halo.<br>Unified Memory · 512k Context · Native Responses API</h2>

<p align="center">
Engineered specifically for the AMD Strix Halo unified memory architecture.<br>
Designed for heavy concurrent coding workloads (OpenCode 2, Codex CLI, Cursor) without multi-second decode stalls.
</p>

| Metric | On one AMD Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X) |
|---|---|
| **Multi-Agent Decode (4 slots)** | **104.4 tok/s aggregate mean** (up to **115.5 tok/s** peak) |
| **Solo Decode (T = 0.0)** | **89.7 tok/s** (P2 math) · **86.8 tok/s** (P1 code) · **73.8 tok/s** Grand Mean |
| **Prefill Bandwidth** | **~1,820 tok/s** (Monotonic Peak) · **~1,458 tok/s** (Real-World Code at 32k) |
| **Context Window** | **512k tokens** with YaRN and Q8 chunked KV cache |
| **Bit-Identity Law** | **100.000% exact SHA-256 determinism** across all standardized benchmarks |
| **Agent Protocol** | **OpenAI Responses API (`/v1/responses`)** over HTTP & WebSocket |

---

## Production Verification & Performance

All benchmarks were executed on bare metal running Arch Linux (Kernel 7.3-zen) with recommended kernel headroom (`ttm.pages_limit=31457280 amdgpu.gttsize=122880 amd_iommu=off transparent_hugepage=always`) using **ThinkingCap** weights (`Thinking-Cap-Qwen-3.8-FN`) under the [Recommended Hybrid Q6 Quantization Layout](docs/WEIGHTS.md).

Full methodology, per-prompt latency distributions, and reproducibility scripts: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### 1. Single-Stream (Solo) Performance & Determinism
*1 Active Slot, Greedy Decoding (T = 0.0), n = 5 consecutive sterile trials with `prompt-cache = off` (+ 1 warmup).*

| Task | Domain / Subject | Completion | TTFT (ms) | Decode Throughput | Bit-Identity Status (SHA-256) |
|---|---|---|---|---|---|
| **P1** | Systems & Concurrency (Python 3.12 DAG) | 256 tok | 57.7 ± 14.0 | **86.77 ± 0.03 tok/s** | **100.000% MATCH** (`300ca5...`) |
| **P2** | Formal Mathematics (Riemann-Lebesgue) | 256 tok | 51.9 ± 0.4 | **89.67 ± 0.01 tok/s** | **100.000% MATCH** (`721be7...`) |
| **P3** | Linux Kernel Internals (4-Level Page Walk) | 256 tok | 52.4 ± 0.4 | **77.41 ± 0.06 tok/s** | **100.000% MATCH** (`4f71d1...`) |
| **P4** | Russian Technical Translation & Stylistics | 256 tok | 51.9 ± 0.5 | **67.98 ± 0.01 tok/s** | **100.000% MATCH** (`93837e...`) |
| **P5** | Adversarial Abstention & Tool Dispatch | 133 tok | 51.6 ± 0.6 | **59.94 ± 0.01 tok/s** | **100.000% MATCH** (`5f8ea9...`) |
| **P6** | Agentic Engine Architecture & DRR Arbiter | 256 tok | 52.2 ± 0.6 | **60.89 ± 0.00 tok/s** | **100.000% MATCH** (`1cbb83...`) |
| **MEAN** | **Grand Mean Across All Prompts** | — | **53.0 ms** | **73.78 tok/s** | **100.000% VERIFIED** |

### 2. Prefill Context Scaling
*1 Active Slot, T = 0.0, chunk = 4096: Monotonic compute peak vs Real-world non-repeating C++ source repository files.*

| Context Window | Evaluated Tokens | Monotonic / GEMM Peak | Real Code (Cold NVMe) | Real Code (Warm RAM/PLE) | Real Code TTFT |
|---|---|---|---|---|---|
| **4k Context** | 4,395 tokens | **1,865.0 tok/s** | 1,388.9 tok/s | **1,437.6 ± 30.4 tok/s** (n = 5) | 3.06 s |
| **16k Context** | 15,683 tokens | **1,844.2 tok/s** | 1,433.8 tok/s | **1,454.2 ± 4.5 tok/s** (n = 5) | 10.78 s |
| **32k Context** | 30,899 tokens | **1,819.5 tok/s** | 1,440.9 tok/s | **1,457.9 ± 6.2 tok/s** (n = 5) | 21.20 s |
| **64k Context** | 60,183 tokens | **1,771.8 tok/s** | 1,427.9 tok/s | **1,424.9 ± 5.1 tok/s** (n = 3) | 42.22 s |
| **128k Context** | 126,482 tokens | **1,720.5 tok/s** | 1,376.9 tok/s | **1,377.9 ± 1.4 tok/s** (n = 3) | 91.79 s |

*Note:* Real-world evaluation conducted on non-repeating C++ source code (`serve/*.cpp`, `serve/*.hpp`). Zero throughput penalty between cold NVMe and warm RAM at ≥ 16k context because `set_lookahead` asynchronously prefetches chunk k+1 during chunk k compute.

### 3. Multi-Slot Concurrency Scaling
*Parallel concurrent multi-agent load (T = 0.6, n = 3 measurement trials after warmup).*

| Active Slots | Aggregate Throughput | Per-Slot Throughput | Step Latency p50 | Step Latency p95 | Effective ITL |
|---|---|---|---|---|---|
| **1 Slot** | **76.97 ± 13.57 tok/s** | 76.97 ± 13.57 tok/s | 62.4 ms | 69.5 ms | **13.0 ms** |
| **2 Slots** | **92.97 ± 15.57 tok/s** | 50.12 ± 7.66 tok/s | 81.3 ms | 100.6 ms | **20.0 ms** |
| **4 Slots** | **104.38 ± 10.56 tok/s** | **31.07 ± 2.98 tok/s** | 109.8 ms | 129.4 ms | **32.2 ms** |

*Note on Peak Throughput & Concurrency:* Peak 4-slot aggregate throughput reaches **115.5 tok/s** on structured code with high MTP acceptance (~84%), and averages **104.4 tok/s** across diverse mixed workloads. Per-slot generation stays smooth at **31.1 tok/s** (4 × 31.07 = 124.3 tok/s concurrent instantaneous compute). Effective ITL (13.0–32.2 ms) reflects multi-token speculative emission.

### 4. Mixed Concurrency Isolation (3D + 1P / 2D + 2P / 1D + 3P)
*Mutual overlap contention window, continuous looped decoders, n = 3 trials (`chunk = 4096, chunk-shared = 1024`).*

| Profile | Decode / Slot (tok/s) | Prefill Agg (tok/s) | Step p50 | Step p95 | Max Step ITL | Makespan (16k) | GPU Utilization |
|---|---|---|---|---|---|---|---|
| **3D + 1P** (3 Dec, 1 Pref 16k) | **16.75 ± 0.72 tok/s** | **913.3 ± 35.3 tok/s** | 111.9 ms | 770.7 ms | **872.5 ms** | 20.8 s | **100.5%** |
| **2D + 2P** (2 Dec, 2 Pref 16k) | **18.60 ± 1.00 tok/s** | **991.7 ± 24.6 tok/s** | 89.5 ms | 746.4 ms | **836.1 ms** | 38.3 s | **94.5%** |
| **1D + 3P** (1 Dec, 3 Pref 16k) | **25.52 ± 3.04 tok/s** | **1151.3 ± 94.2 tok/s** | 65.3 ms | 718.8 ms | **2307.3 ms** | 49.7 s | **96.4%** |

*Note on Mixed Latency & Arbiter Physics:*
1. **Steady-State Ceiling (< 880 ms):** In 3D+1P and 2D+2P, decode step latency is strictly bounded below **872.5 ms**, proving the DRR GpuArbiter yields after every single 1024-token prefill quantum rather than chaining prefill passes consecutively.
2. **Solo Quantum Physics (~2.3 s):** In 1D+3P (a ~50s prefill window), the single decoder finishes and re-opens requests. In the brief inter-query socket transition, an isolated prefill slot executes a single solo `chunk = 4096` (4096 / 1820 ≈ 2.25 s + step ≈ 2.3 s) before the arbiter detects concurrency and transitions to `chunk-shared = 1024`.
3. **MoE Silicon Efficiency:** `chunk-shared = 1024` avoids the catastrophic memory-bound collapse observed on `chunk-shared = 256` (M ≈ 5 tokens/expert dropping prefill speed to 339 tok/s) while guaranteeing 95–100% total hardware utilization.

---

## Unified Memory Architecture (128 GB DRAM Allocation)

On AMD Strix Halo, the Zen 5 CPU and RDNA 3.5 GPU share physical LPDDR5X-8533 unified memory (~273 GB/s peak). Memory allocations are statically planned and verified at engine startup:

| Memory Component | Allocation | Share of RAM | Engineering Function |
|---|---|---|---|
| **Model Weights (`weights.faf`)** | 65.5 GiB | 51.2% | Resident in GPU-accessible DRAM ([format specification](docs/WEIGHTS.md)) |
| **KV Cache (4 slots × 262,144 tokens)** | 20.4 GiB | 15.9% | Chunked Q8 PagedAttention KV buffers (1.05M total context) |
| **Prompt Cache RAM Tier (Hot Snapshots)** | 22.0 GiB | 17.2% | In-memory KV snapshot tier for instant multi-turn agent switching ([config](deploy/fastasfuck-server.conf)) |
| **Turn Resume Points & Delta History** | 2.2 GiB | 1.7% | 32 in-RAM turn checkpoints (~0.11 GiB each) + deltas for agent backtrack/edits |
| **PLE NVMe Streamer Row Cache** | 1.5 GiB | 1.2% | 8,388,608 dynamic n-gram hot rows cached in RAM |
| **Working Scratch & Activation Buffers** | 0.4 GiB | 0.3% | HIP streams, GEMM workspaces, and WMMA accumulators |
| **Kernel Headroom & Linux OS Safety Margin** | 16.0 GiB | 12.5% | Protected headroom for Linux kernel slabs, TCP sockets, and direct reclaim safety |
| **Total Physical DRAM** | **128.0 GiB** | **100.0%** | **Guaranteed zero direct memory reclaim storms** |

---

## Features

- **OpenAI Responses API (`/v1/responses`)** — HTTP (JSON/SSE) and WebSocket routes with stateful sessions (`previous_response_id`) for zero-overhead multi-turn agent turns.
- **OpenAI-Compatible Chat Server** — `/v1/chat/completions` with streaming, function calling, reasoning, and structured JSON output.
- **Drop-in `.strixw` Compatibility** — natively loads legacy `.strixw` weight packages and tables out of the box with zero reconversion needed.
- **Deficit Round Robin (DRR) GpuArbiter** — quantum prefill slicing (`chunk-shared`) guarantees non-blocking decode streaming during heavy repository prefill.
- **Multi-Token Prediction (MTP) & Two-Stage Act-PCA (`draft2s` / `lm2s`)** — native decoupled draft heads with escalating margin gates, n-gram table assist, and default-enabled two-stage Act-PCA vocabulary heads (`--draft-head-2stage`, `--lm-head-2stage`).
- **512k Context with YaRN** — chunked Q8 KV cache keeps memory flat and attention fast at depth.

| Option | Default | Description |
| :--- | :---: | :--- |
| `--draft-head-2stage` | `on` | Two-Stage Act-PCA draft head (`R=512, M=512`, exact Q8 gather). **100% lossless** (`KLD = 0.0000`) — verified by the exact main head. Cuts draft step time ~30–44% and saves ~126 MB VRAM vs legacy Q4 draft (`en+ru`). |
| `--lm-head-2stage` | `on` | Two-Stage Act-PCA verify LM head (`R=512, M=1536`, exact Q8 gather over the full 248k vocab). Saves 2.2–2.7 ms per verify round (+6–12% end-to-end tok/s). Auto-falls back to full Q8 for `top_k=0`, `top_k > 20`, grammars/structured output, `logit_bias`, and FP32. |

### Two-Stage Act-PCA Vocabulary Heads (`--draft-head-2stage`, `--lm-head-2stage`)

On Strix Halo (32 MiB MALL, 273 GB/s LPDDR5X), dense 248k-vocab projections (`676 MB` in Q8) thrash cache on every decode/verify step. Both heads are enabled by default and run a 3-stage GPU pipeline: project hidden states to a 512-dim Activation-PCA subspace, shortlist top candidates in Q4 (`35.3 MB` draft for `en+ru` / `74.1 MB` LM head — less memory combined than the legacy 1.2.2 Q4 draft head alone), then compute **exact Q8 dot products** on the shortlisted rows.

* **First startup:** runs a one-time ~5 s calibration and caches the orthonormal basis in `draft2s_pca_r512.bin` (~2.6 MB) next to `weights.faf` (the weights directory should be writable on first launch; otherwise the basis recomputes cleanly at startup). Subsequent starts reload from cache in ~0.2 s, and `mtp_calib_cache.json` profiles isolate automatically (`+2s_r512_m512`, `+lm2s_m1536`).
* **Accuracy & safety:**
  * `--draft-head-2stage` is **mathematically lossless** (`KLD = 0.0000`; the verifier rejects and resamples any draft miss cleanly) and raises draft Top-1 agreement vs. Q8 from 96.5% to 100%.
  * `--lm-head-2stage` is **bit-identical at `T = 0.0`** (6/6 SHA-256 parity vs. full Q8) and virtually exact on live decode (`0/936` misses at `T = 0.6`, `KLD ~ 3e-13`; `0/128` on `<think>` at `T = 1.0`, `KLD ~ 1.4e-13`).
  * **Why `KLD ~ 3e-13` is physical zero (not a math/WMMA error):** Stage 3 computes the exact same asymmetric Q8 `dot2` arithmetic against `lm_head.weight` as full Q8 — **not** WMMA (which would cause `~1e-3` TVD / `~1e-6` KLD). Because FP32 FMA reduction order differs slightly between the gather kernel and full-matrix kernel, logits shift by at most ~1 ULP in FP32 (`~1.19e-7`, giving `TVD ~ 1e-7`), and quadratic KL expansion near `q = p` yields `KLD ~ (1e-7)^2 ~ 1e-14..3e-13` (scaling by `(1/T)^2` between `T = 1.0` and `T = 0.6`).
  * **Automatic fallback:** any request using `top_k = 0`, `top_k > 20`, JSON schema/grammar masks, `logit_bias`, or FP32 automatically bypasses `lm_head_2s` and runs the full Q8 `lm_head`.

**When to turn them off:**
* **`--lm-head-2stage off`**: disable if you run high-temperature (`T >= 1.0`) creative writing in CJK or rare scripts with `top_k <= 20` and want zero tail pruning, if you run strict reference perplexity/KLD evals against full Q8, or if all your traffic hits the auto-fallback path (`top_k = 0` / grammars) and you want to reclaim 74 MB of VRAM.
* **`--draft-head-2stage off`**: rarely needed — keep it on unless you are A/B-benchmarking against the legacy Q4 draft head.

For full details, see Section 5 of [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## OpenAI Responses API (HTTP & WebSocket)

fastasfuck natively supports the OpenAI Responses API (`/v1/responses`) over both HTTP (JSON/SSE) and WebSocket for low-latency coding agents like **Codex CLI** and **OpenCode 2**:

- **Incremental state with `previous_response_id`**: upload history only once. Subsequent turns send only new deltas (~10–30 tokens), reusing 100% of the live session KV cache on the GPU without redundant tokenization or prefill:
  - **Cold Turn (Turn 1):** Ingests 15,000 tokens of file contents (~9.6s prefill).
  - **Subsequent Turns (Turns 2..N):** Server matches live GPU session and prefills **only the 20–50 token delta in < 25 ms** (TTFT reduced by 99.8%).
- **WebSocket Route (`ws://localhost:5300/v1/responses`)**: maintains an active multi-turn connection, streaming granular token events (`response.output_text.delta`, `response.reasoning_text.delta`, `response.function_call_arguments.delta`) and handling non-blocking cancellation (`response.cancel`).
- **Reasoning Control**: control think blocks dynamically with `reasoning: {"effort": "none" | "low" | "medium" | "high"}`.
- **Flat & Nested Tool Calls**: natively handles flat and nested tool schemas with seamless `function_call_output` chaining.

---

## Quick Start

You need an AMD Strix Halo machine with 128 GB of memory running Arch Linux (or compatible Linux distribution), and a fast NVMe drive with ~120 GB free for the weights and n-gram table.

### 1. Kernel Command Line Configuration
To prevent ROCm page-fault stalls and ensure safe allocation headroom, add the following parameters to your bootloader (e.g. in `/etc/default/grub` or systemd-boot loader entry) and reboot:

```bash
ttm.pages_limit=31457280 amdgpu.gttsize=122880 amd_iommu=off transparent_hugepage=always
```

### 2. Install Native Arch ROCm Packages & Build
fastasfuck builds natively on Arch Linux using official ROCm packages for `gfx1151` (tested with ROCm 10.1 / HIP 7.16):

```sh
# Install native Arch ROCm packages
sudo pacman -S rocm-hip-sdk hip-runtime-amd rocm-llvm cmake ninja

git clone https://github.com/Cheblan/fastasfuck && cd fastasfuck
cmake --preset strix && cmake --build --preset strix
```

### 3. Download Quantized Model Weights
Download pre-quantized weights, n-gram table, and tokenizer from the [Hugging Face Collection](https://huggingface.co/collections/Cheblan/fastasfuck-quantizations):

```sh
# Example: ThinkingCap (recommended production default)
huggingface-cli download Cheblan/ThinkingCap-Qwen3.8-Flash-Next-faf --local-dir /models/Thinking-Cap-Qwen-3.8-FN-faf
```

### 4. Run the Server
Serve an OpenAI-compatible API on port 5300:

```sh
build/strix/fastasfuck_server --config deploy/fastasfuck-server.conf
```

For a comprehensive guide to every configuration parameter, memory sizing, and production recommendations, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### 5. Verify & Query
```sh
curl -s localhost:5300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Point any OpenAI-compatible client (Codex CLI, OpenCode 2, Cursor) at `http://<your-machine>:5300/v1`, using the model ID returned by `/v1/models`.

---

## Quantized Weights & Custom Calibration (FP64 ActOrder GPTQ)

Production `.faf` packages pair Hybrid `Q4_G128` (`exp_gu`, `exp_down`, `gdn_in`) + `Q6_G64` (dense attention/GDN out/shared expert) + `Q8_G64` (`hc`, `lm_head`) with **full-rank 61,440-token FP64 ActOrder GPTQ (`static_groups=True` + Empirical Bayes Shrink512 across all 512 experts)**:

- **`-30.7%` lower held-out KLD vs. RTN** (`0.031249 → 0.021643`, Top-1 vs BF16 `94.80% → 95.85%`, Top-5 `99.94%`).
- **Exact byte & speed parity:** `65.51 GiB` (`70,337,409,024 B`), zero `g_idx` overhead (`74.16 ± 2.04 tok/s` Solo MTP across 30 prompts at `T = 0.0` vs `73.65` RTN; `71.81` vs `68.43 tok/s` at `T = 0.6`).
- **Quantize custom checkpoints or evaluate GPTQ vs GPTAQ:** Run [`tools/quantize_qwen_gptq.py`](tools/quantize_qwen_gptq.py) + `convert_qwen4exp --prebuilt`. See [docs/WEIGHTS.md](docs/WEIGHTS.md) for the complete `.faf` precision schema, GPTQ vs. GPTAQ ablation findings, and ROCm/Strix Halo quantization best practices.

---

## License

[AGPL-3.0](LICENSE). The engine server is licensed under AGPL-3.0. Licenses for served model weights are separate and governed by their respective authors.
