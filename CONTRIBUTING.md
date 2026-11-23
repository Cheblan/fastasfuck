# Contributing to fastasfuck

Thank you for your interest in contributing to **fastasfuck**, a high-throughput, production-grade LLM inference engine engineered specifically for the AMD Strix Halo unified memory architecture (`gfx1151`).

We welcome contributions from the community — bug fixes, kernel optimizations, documentation improvements, and architectural features. All contributions are accepted under the project's license, [AGPL-3.0](LICENSE).

To maintain high engine stability and low-latency serving reliability, contributions are organized into two distinct tracks: **Code & Architecture** and **Documentation & Guides**.

---

## 1. Code & Architecture Contributions

The engine serves concurrent multi-agent production workloads under strict latency and memory budgets. Any modifications to C++, HIP, kernels, memory allocators, or scheduling must satisfy the following engineering standards:

### A. The Bit-Identity Law (100.000% Determinism at T=0)
- Greedy decoding (`temperature = 0.0`, `top_p = 1.0`, `top_k = 1`) is mathematically deterministic.
- Any change to compute kernels (WMMA GEMM, GDN linear attention, QSA windowed attention, MoE routing, MTP draft heads), memory layout, or numerical precision must produce **100.000% bit-identical token output** matching the reference baseline.
- Even a single token discrepancy or numerical drift at T=0 renders a PR invalid.

### B. Tiered Verification & Benchmarking Standard (n=5 / n=3)
Performance claims must be substantiated with reproducible, multi-trial measurements on bare-metal AMD Strix Halo silicon:
1. **Mandatory Untimed Warmup (Iteration 0):** Every benchmark run must first execute an untimed warmup request (hardware ping / allocator initialization). Iteration 0 timing is discarded from all statistics.
2. **Solo & Short Prefill (n=5 Trials):** For single-slot decode (standardized prompts P1–P6) and short prefill sweeps (4k–32k), execute 5 consecutive identical trials with `prompt-cache = off`. Report `mean ± standard deviation`.
3. **Multi-Slot & Mixed Concurrency (n=3 Trials):** For multi-slot suites (4 slots) and mixed concurrency profiles (3D + 1P, 2D + 2P), execute 3 full measurement trials after warmup.
4. **Real Workload Rule:** When testing multi-slot concurrency, every active slot must execute real, concurrent generation passes under genuine load (synthetic empty loops or sleep calls are prohibited).

### C. GpuArbiter & UMA Memory Safety
- On Unified Memory Architecture (UMA), CPU memory pressure or aggressive page allocation directly starves GPU memory controllers.
- If modifying `GpuArbiter`, memory allocators, or KV buffers, verify:
  - Zero memory leaks and zero allocator deadlocks under sustained multi-agent concurrency.
  - Sliced prefill (`chunk-shared`) yields cooperatively to active decode slots, guaranteeing bounded inter-token latency.
  - Total static memory allocation stays within the 128 GB DRAM budget, preserving the protected OS and kernel headroom (zero direct memory reclaim storms).

### D. Code Quality & Standards
- Modern, idiomatic C++20 and HIP.
- Clean kernel design: explicit thread-block tiling, Wave32 execution alignment, and shared-memory bank conflict avoidance.
- Small, focused pull requests targeting a single optimization or bug fix are preferred.

---

## 2. Documentation & Guide Contributions

Clear, well-crafted documentation is just as essential as fast kernels. High-quality documentation ensures that operators, developers, and researchers can reliably deploy the engine, understand its memory architecture, and reproduce performance results.

### A. Readability & Visual Aesthetics
- **Clean Structure:** Organize documents with clear hierarchical headings, concise paragraphs, and generous whitespace. Avoid monolithic blocks of dense, unformatted text.
- **Eye-Friendly Markdown:** Use GitHub Flavored Markdown (GFM) tables, formatted callouts, and fenced code blocks with appropriate syntax highlighting (`bash`, `cpp`, `ini`, `json`, `mermaid`).
- **Clean Metrics & Formatting:** Prefer Unicode characters (`±`, `≈`, `≥`, `×`, `—`) and plain-text units (`tok/s`, `ms`, `s`, `GiB`) over brittle inline math blocks that break rendering across web and mobile interfaces. Reserve display LaTeX math (`$$...$$`) for complex mathematical derivations.

### B. Technical Precision & Ground Truth
- **Grounded in Reality:** Documentation must strictly reflect the actual codebase, physical hardware limits, and verified benchmark data. Never cite theoretical peak numbers without disclosing operational parameters (e.g. Multi-Token Prediction acceptance rates, context lengths, chunk sizes).
- **Consistent Terminology:**
  - Project name: **fastasfuck**
  - Weight format: **`.faf`**
  - Target hardware: **AMD Strix Halo (`gfx1151`, Ryzen AI Max+ 395, 128 GB LPDDR5X-8533)**
  - Baseline environment: **Arch Linux (Linux Kernel 7.3-zen, native `/opt/rocm` packages)**
- **Actionable & Reproducible:** Setup commands, configuration snippets, and build instructions must be copy-paste ready and tested against clean environments.

---

## 3. How to Submit a Pull Request

1. **Fork & Branch:** Fork the repository on GitHub and create a dedicated branch (e.g., `feat/opt-swiglu` or `docs/update-weights-spec`).
2. **Build & Test:**
   - For code: Ensure a clean compilation with CMake (`cmake --preset strix && cmake --build --preset strix`) and run verification tests.
   - For docs: Preview markdown rendering to verify formatting, tables, and link integrity.
3. **Open a PR:** Target `main` with a clear description of your changes:
   - For performance PRs: Include before/after benchmark results or JSON output from `tools/bench/`.
   - For bug fixes: Include reproduction steps or minimal curl payloads.
4. **Review & Merge:** Once validated on AMD Strix Halo hardware, your contribution will be merged directly into `main` with full git author attribution.
