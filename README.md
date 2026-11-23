<p align="center"><img src="assets/strixite-banner.jpg" alt="strixite - LLM inference, from scratch, for AMD Strix Halo" width="100%"></p>

<p align="center">
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-6f42c1"></a>
  <img alt="Platform: Linux, AMD Strix Halo (gfx1151)" src="https://img.shields.io/badge/platform-Linux%20%C2%B7%20Strix%20Halo%20gfx1151-1abc9c">
  <a href="https://huggingface.co/wemoh/Qwen3.8-Flash-Next-strixw"><img alt="Weights on Hugging Face" src="https://img.shields.io/badge/weights-Hugging%20Face-ffcc4d"></a>
</p>

<h2 align="center">High-Throughput Multi-Agent LLM Engine for AMD Strix Halo.<br>Unified Memory · 512k Context · Native Responses API</h2>

<p align="center">
Engineered specifically for the AMD Strix Halo unified memory architecture.<br>
Designed for heavy concurrent coding workloads (OpenCode 2, Codex CLI, Cursor) without multi-second decode stalls.
</p>

| Metric | On one AMD Strix Halo (Ryzen AI Max+ 395, 128 GB LPDDR5X) |
|---|---|
| **Multi-Agent Decode (4 slots)** | **104.4 tok/s aggregate mean** (up to **115.5 tok/s** peak) |
| **Solo Decode ($T=0.0$)** | **89.7 tok/s** (P2 math) · **86.8 tok/s** (P1 code) · **73.8 tok/s** Grand Mean |
| **Prefill Bandwidth** | **~1,458 tok/s** at 32k · **~1,378 tok/s** at 128k context |
| **Context Window** | **512k tokens** with YaRN and Q8 chunked KV cache |
| **Bit-Identity Law** | **100.000% exact SHA-256 determinism** across all standardized benchmarks |
| **Agent Protocol** | **OpenAI Responses API (`/v1/responses`)** over HTTP & WebSocket |

---

## Production Verification & Performance

All benchmarks were executed on bare metal running Linux 6.13 with recommended kernel headroom (`ttm.pages_limit=31457280 amdgpu.gttsize=122880 amd_iommu=off transparent_hugepage=always`) under the [Recommended Hybrid Q6 Quantization Layout](docs/weights.md).

Full methodology, per-prompt latency distributions, and reproducibility scripts: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### 1. Single-Stream (Solo) Performance & Determinism
*1 Active Slot, Greedy Decoding ($T=0.0$), $n=5$ consecutive sterile trials with `prompt-cache = off` (+ 1 warmup).*

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
*1 Active Slot, $T=0.0$, evaluated on non-repeating C++ source repository files, cold NVMe lookahead reads vs warm RAM/PLE cache.*

| Context Window | Evaluated Tokens | Cold Prefill (NVMe) | Warm Prefill (RAM/PLE) | Time to First Token (TTFT) |
|---|---|---|---|---|
| **4k Context** | 4,395 tokens | 1,388.9 tok/s | **1,437.6 ± 30.4 tok/s** ($n=5$) | 3.06 s |
| **16k Context** | 15,683 tokens | 1,433.8 tok/s | **1,454.2 ± 4.5 tok/s** ($n=5$) | 10.78 s |
| **32k Context** | 30,899 tokens | 1,440.9 tok/s | **1,457.9 ± 6.2 tok/s** ($n=5$) | 21.20 s |
| **64k Context** | 60,183 tokens | 1,427.9 tok/s | **1,424.9 ± 5.1 tok/s** ($n=3$) | 42.22 s |
| **128k Context** | 126,482 tokens | 1,376.9 tok/s | **1,377.9 ± 1.4 tok/s** ($n=3$) | 91.79 s |

*Note:* Evaluated on non-repeating C++ source code (`serve/*.cpp`, `serve/*.hpp`). Zero throughput penalty between cold NVMe and warm RAM at $\ge 16\text{k}$ because `set_lookahead` asynchronously prefetches chunk $k+1$ during chunk $k$ compute.

### 3. Multi-Slot Concurrency Scaling
*Parallel concurrent multi-agent load ($T=0.6$, $n=3$ measurement trials after warmup).*

| Active Slots | Aggregate Throughput | Per-Slot Throughput | Step Latency p50 | Step Latency p95 | Effective ITL |
|---|---|---|---|---|---|
| **1 Slot** | **76.97 ± 13.57 tok/s** | 76.97 ± 13.57 tok/s | 62.4 ms | 69.5 ms | **13.0 ms** |
| **2 Slots** | **92.97 ± 15.57 tok/s** | 50.12 ± 7.66 tok/s | 81.3 ms | 100.6 ms | **20.0 ms** |
| **4 Slots** | **104.38 ± 10.56 tok/s** | **31.07 ± 2.98 tok/s** | 109.8 ms | 129.4 ms | **32.2 ms** |

*Note on Peak Throughput & Concurrency:* Peak 4-slot aggregate throughput reaches **115.5 tok/s** on structured code with high MTP acceptance (~84%), and averages **104.4 tok/s** across diverse mixed workloads. Per-slot generation stays smooth at **31.1 tok/s** ($4 \times 31.07 = 124.3\text{ tok/s}$ concurrent instantaneous compute). Effective ITL ($13.0 - 32.2\text{ ms}$) reflects multi-token speculative emission.

### 4. Mixed Concurrency Isolation (3D + 1P / 2D + 2P / 1D + 3P)
*Mutual overlap contention window, continuous looped decoders, $n=3$ trials (`chunk = 4096, chunk-shared = 1024`).*

| Profile | Decode / Slot (tok/s) | Prefill Agg (tok/s) | Step p50 | Step p95 | Max Step ITL | Makespan (16k) | GPU Utilization |
|---|---|---|---|---|---|---|---|
| **3D + 1P** (3 Dec, 1 Pref 16k) | **16.75 ± 0.72 tok/s** | **913.3 ± 35.3 tok/s** | 111.9 ms | 770.7 ms | **872.5 ms** | 20.8 s | **100.5%** |
| **2D + 2P** (2 Dec, 2 Pref 16k) | **18.60 ± 1.00 tok/s** | **991.7 ± 24.6 tok/s** | 89.5 ms | 746.4 ms | **836.1 ms** | 38.3 s | **94.5%** |
| **1D + 3P** (1 Dec, 3 Pref 16k) | **25.52 ± 3.04 tok/s** | **1151.3 ± 94.2 tok/s** | 65.3 ms | 718.8 ms | **2307.3 ms** | 49.7 s | **96.4%** |

*Note on Mixed Latency & Arbiter Physics:*
1. **Steady-State Ceiling ($< 880\text{ ms}$):** In 3D+1P and 2D+2P, decode step latency is strictly bounded below **872.5 ms**, proving the DRR GpuArbiter yields after every single 1024-token prefill quantum rather than chaining prefill passes consecutively.
2. **Solo Quantum Physics ($2.3\text{ s}$):** In 1D+3P (a ~50s prefill window), the single decoder finishes and re-opens requests. In the brief inter-query socket transition, an isolated prefill slot executes a single solo `chunk = 4096` ($4096 / 1820 \approx 2.25\text{ s} + \text{step} \approx 2.3\text{ s}$) before the arbiter detects concurrency and transitions to `chunk-shared = 1024`.
3. **MoE Silicon Efficiency:** `chunk-shared = 1024` avoids the catastrophic memory-bound collapse observed on `chunk-shared = 256` ($M \approx 5$ tokens/expert dropping prefill speed to $339\text{ tok/s}$) while guaranteeing 95–100% total hardware utilization.

---

## Unified Memory Architecture (128 GB DRAM Allocation)

On AMD Strix Halo, the Zen 5 CPU and RDNA 3.5 GPU share physical LPDDR5X-8533 unified memory (~273 GB/s peak). Memory allocations are statically planned and verified at engine startup:

| Memory Component | Allocation | Share of RAM | Engineering Function |
|---|---|---|---|
| **Model Weights (`weights-q6-hybrid.strixw`)** | 65.5 GiB | 51.2% | Resident in GPU-accessible DRAM ([what quant is this?](docs/weights.md)) |
| **KV Cache (4 slots × 262,144 tokens)** | 20.4 GiB | 15.9% | Chunked Q8 PagedAttention KV buffers ($1.05\text{M}$ total context) |
| **PLE NVMe Streamer Row Cache** | 1.5 GiB | 1.2% | 8,388,608 dynamic n-gram hot rows cached in RAM |
| **Working Scratch & Activation Buffers** | 0.4 GiB | 0.3% | HIP streams, GEMM workspaces, and WMMA accumulators |
| **Kernel Headroom & Linux OS Safety Margin** | 40.2 GiB | 31.4% | Protected headroom for Linux kernel slabs, TCP sockets, and zero-copy user buffers |
| **Total Physical DRAM** | **128.0 GiB** | **100.0%** | **Guaranteed zero direct memory reclaim storms** |

---

## Features

- **OpenAI Responses API (`/v1/responses`)** — HTTP (JSON/SSE) and WebSocket routes with stateful sessions (`previous_response_id`) for zero-overhead multi-turn agent turns ([docs](docs/server.md)).
- **OpenAI-Compatible Chat Server** — `/v1/chat/completions` with streaming, function calling, reasoning, and structured JSON output.
- **Deficit Round Robin (DRR) GpuArbiter** — quantum prefill slicing (`chunk-shared`) guarantees non-blocking decode streaming during heavy repository prefill.
- **Multi-Token Prediction (MTP)** — native decoupled draft heads with escalating margin gates and n-gram table assist.
- **Images** — PNG, JPEG, WebP, GIF, BMP, and TGA in chat requests via separate vision tower ([images](docs/images.md)).
- **512k Context with YaRN** — chunked Q8 KV cache keeps memory flat and attention fast at depth.

---

## OpenAI Responses API (HTTP & WebSocket)

strixite natively supports the OpenAI Responses API (`/v1/responses`) over both HTTP (JSON/SSE) and WebSocket for low-latency coding agents like **Codex CLI** and **OpenCode 2**:

- **Incremental state with `previous_response_id`**: upload history only once. Subsequent turns send only new deltas (~10–30 tokens), reusing 100% of the live session KV cache on the GPU without redundant tokenization or prefill:
  - **Cold Turn (Turn 1):** Ingests 15,000 tokens of file contents (~9.6s prefill).
  - **Subsequent Turns (Turns 2..N):** Server matches live GPU session and prefills **only the 20–50 token delta in < 25 ms** (TTFT reduced by 99.8%).
- **WebSocket Route (`ws://localhost:5300/v1/responses`)**: maintains an active multi-turn connection, streaming granular token events (`response.output_text.delta`, `response.reasoning_text.delta`, `response.function_call_arguments.delta`) and handling non-blocking cancellation (`response.cancel`).
- **Reasoning Control**: control think blocks dynamically with `reasoning: {"effort": "none" | "low" | "medium" | "high"}`.
- **Flat & Nested Tool Calls**: natively handles flat and nested tool schemas with seamless `function_call_output` chaining.

---

## Quick Start

You need an AMD Strix Halo machine with 128 GB of memory running Linux, and a fast NVMe drive with ~120 GB free for the weights and n-gram table.

### 1. Kernel Command Line Configuration
To prevent ROCm page-fault stalls and ensure safe allocation headroom, add the following parameters to your bootloader (on Fedora: `sudo grubby --update-kernel=ALL --args="..."`) and reboot:

```bash
ttm.pages_limit=31457280 amdgpu.gttsize=122880 amd_iommu=off transparent_hugepage=always
```

### 2. Install Toolchain & Build
strixite builds with AMD TheRock ROCm for gfx1151 (tested with 10.1.0 GA / HIP 7.16):

```sh
git clone https://github.com/strixite/strixite && cd strixite
cmake --preset strix && cmake --build --preset strix
```

### 3. Run the Server
Serve an OpenAI-compatible API on port 5300:

```sh
build/strix/strix_server --config deploy/strix-server.conf
```

### 4. Verify & Query
```sh
curl -s localhost:5300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Point any OpenAI-compatible client (Codex CLI, OpenCode 2, Cursor) at `http://<your-machine>:5300/v1`, using the model ID returned by `/v1/models`.

---

## License

[AGPL-3.0](LICENSE). The engine server is licensed under AGPL-3.0. Licenses for served model weights are separate and governed by their respective authors.
