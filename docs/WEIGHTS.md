# FASTASFUCK Model Weights Specification (.faf)

This document specifies the exact quantization geometry, memory layout, and tensor formats for the **fastasfuck** inference engine on AMD Strix Halo silicon (Ryzen AI Max+ 395, `gfx1151`, 128 GB Unified LPDDR5X-8533).

---

## 1. Executive Summary: The Hybrid Q6 / Q4 / Q8 Architecture

- **Weights Container Format:** `.faf` (FastAsFuck unified memory tensor package). Full drop-in compatibility: natively loads legacy `.strixw` weight files and tables without reconversion.
- **Target Baseline Model:** Qwen 3.8-Flash-Next (128.4B total MoE parameters / ~6B activated per token, 48 hybrid GDN/QSA layers).
- **Total In-Memory Footprint:** **65.5 GiB** (70,337,409,024 bytes) resident in GPU-accessible unified DRAM.
- **Routed Experts (MoE):** **`q4g128`** across 512 fine-grained experts (top-10 routed per token) for maximum bandwidth efficiency.
- **Dense Projections & Shared Expert:** **`q6g64`** preserving critical representational quality on frequent execution paths.
- **Sensitive Projections & Residual Mixers:** **`q8g64`** for hyper-connections (`hc_down`, `hc_up`) and LM head (`lm_head`).
- **Embeddings & Router:** Native **BF16** (unquantized) to eliminate lexical drift and routing mispredictions.
- **MTP Speculative Heads:** **`q4g64`** for MTP FC projections (`fc_embedding`, `fc_hidden`).
- **N-gram Speculative Table:** 8-bit quantized on-disk NVMe table (~48.3 GiB) with dynamic RAM caching (`ngram-cache-rows = 8388608`).

---

## 2. Production Layout Specification (14-Field Schema)

The production model file (`weights-q6-hybrid.faf`) is built according to the 14-field layout configuration:

```text
layout = exp_gu=q4g128,exp_down=q4g128,sh_gu=q6g64,sh_down=q6g64,gdn_in=q4g128,gdn_out=q6g64,attn_qkv=q6g64,attn_o=q6g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q6g64,lm_head=q8g64,mtp_fc=q4g64,embed=bf16
```

### Exact Precision & Tensor Breakdown

| Layout Field | Sub-Module / Tensor Description | Precision Format | Scale Granularity | Target Silicon Execution |
|---|---|---|---|---|
| **`exp_gu`** | Routed MoE Experts: Gate & Up Projections | **`q4g128`** | Group 128 | Dual SIMD32 WMMA (Wave32) |
| **`exp_down`** | Routed MoE Experts: Down Projections | **`q4g128`** | Group 128 | Dual SIMD32 WMMA (Wave32) |
| **`sh_gu`** | Shared Expert: Gate & Up Projections | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`sh_down`** | Shared Expert: Down Projections | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`gdn_in`** | Linear Attention (GDN): Input Projections | **`q4g128`** | Group 128 | Dual SIMD32 WMMA (Wave32) |
| **`gdn_out`** | Linear Attention (GDN): Output Projections | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`attn_qkv`** | Full Window Attention: Q, K, V Projections | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`attn_o`** | Full Window Attention: Output Projections | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`hc_down`** | Hyper-Connection Mixer: Down Projection | **`q8g64`** | Group 64 | Dot2 Vectorized Accumulator |
| **`hc_up`** | Hyper-Connection Mixer: Up Projection | **`q8g64`** (chunk-major) | Group 64 | Dot2 Vectorized Accumulator |
| **`ple_kv`** | Prompt-Lookup Engine: KV Projection | **`q6g64`** | Group 64 | WMMA FP16 Accumulator |
| **`lm_head`** | Language Model Head: Final Unembedding | **`q8g64`** | Group 64 | High-Precision Dot2 / FP32 |
| **`mtp_fc`** | MTP Heads: Embedding & Hidden FC Layers | **`q4g64`** | Group 64 | Draft WMMA Acceleration |
| **`embed`** | Token Embedding Matrix (`embed_tokens`) | **BF16** | Unquantized | Zero Lexical Quantization Error |

In addition to the 14 layout fields:
- **MoE Router (`router`):** Native **BF16** (unquantized) to eliminate gating score errors.
- **RMSNorms, Residual Scales, Conv1D Biases:** Native **FP32** channels.
- **Total Weights Footprint:** **65.5 GiB** resident in unified memory.

---

## 3. Quantization Mechanics: The Hybrid Rationale

1. **MoE Sparse Throughput (`q4g128`):**  
   With 512 experts and top-k = 10 routing per token, memory bandwidth is the primary bottleneck during multi-token decode steps. Using `q4g128` on routed expert weights cuts memory bus traffic by over 50% compared to 8-bit, enabling aggregate multi-slot throughput exceeding 104 tok/s on AMD Strix Halo's 273 GB/s memory subsystem.

2. **Dense Path Accuracy (`q6g64`):**  
   Every token must pass through the shared expert, GDN projections, and attention blocks. Quantizing these dense layers to `q6g64` prevents the error compounding that occurs when dense layers are aggressively quantized to 4-bit, maintaining near-BF16 reasoning and coding accuracy.

3. **Critical Invariants (`q8g64` & `BF16`):**  
   Hyper-connection residual mixings and the LM head are kept in high-precision `q8g64` to prevent logits collapse and preserve output calibration across extended agent trajectories.

---

## 4. Serving Configuration

In `deploy/fastasfuck-server.conf`:

```ini
# Path to packed .faf weights
weights = /models/Thinking-Cap-Qwen-3.8-FN-faf/weights-q6-hybrid.faf

# Dedicated n-gram streaming table on NVMe
ngram = /models/Thinking-Cap-Qwen-3.8-FN-faf/ngram.table

# Tokenizer and generation metadata
tokenizer = /models/Thinking-Cap-Qwen-3.8-FN-faf/tokenizer.json
generation-config = /models/Thinking-Cap-Qwen-3.8-FN-faf/generation_config.json
```
