# Server Configuration Guide (`fastasfuck-server.conf`)

This manual provides an in-depth reference for configuring the **fastasfuck** high-throughput LLM inference engine on AMD Strix Halo unified memory architecture (gfx1151 / Zen 5, 128 GB LPDDR5X-8533).

---

## 1. Syntax, Precedence, and Validation Rules

### 1.1 File Syntax
- **Format:** Key-value pairs structured as `key = value` (one setting per line).
- **Whitespace:** Whitespace around `=` is ignored (`chunk = 4096` is identical to `chunk=4096`).
- **Comments:** The `#` character marks comments anywhere on a line. Empty or whitespace-only lines are ignored.
- **Paths:** All filesystem paths defined in configuration files **must be absolute**. Relative paths depend on the invoker's working directory and are rejected at startup.
- **Booleans:** Flags take strictly `on` or `off`. Values such as `true`, `false`, `1`, or `0` are rejected by the strict parser.

### 1.2 Configuration Precedence
Engine configuration follows a strict three-tier precedence model:

```text
Built-in C++ Defaults < Configuration File (--config PATH) < CLI Flags (--key value)
```

1. **Built-in Defaults:** Compiled into `fastasfuck_server_options()` (`serve/server_config.cpp`).
2. **Configuration File (`--config PATH`):** Overrides compiled defaults. If `--config` is omitted, the server operates on built-in defaults.
3. **Command-Line Flags (`--key value`):** Possess the highest precedence and override both configuration file entries and defaults.

### 1.3 Strict Validation Invariant
The engine **never silently ignores unknown or invalid configuration options**. Any configuration defect causes an immediate, fatal startup abort naming the offending file, line number, or CLI flag:
- Unknown keys (protects against typos such as `total_slots` instead of `total-slots`).
- Duplicate keys in the file or across command-line flags.
- Lines missing an `=` delimiter or defining empty values.
- Type mismatches or numbers outside their physical allowed ranges `[lo, hi]`.
- Unreadable or missing configuration files.

---

## 2. Comprehensive Configuration Reference

### 2.1 Model Files & Binary Formats

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `weights` | `Path` | Valid file path | `/models/weights.faf` | Absolute path to quantized base model weights in `.faf` binary format (FastAsFuck unified tensor layout). |
| `ngram` | `Path` | Valid file path | `/models/ngram.table` | Absolute path to the prefix n-gram lookup table (int8 rows) powering the Prefix Lookahead Engine (PLE). |
| `tokenizer` | `Path` | Valid file path | `/models/tokenizer.json` | Absolute path to the HuggingFace `tokenizer.json` vocabulary file. |
| `generation-config` | `Path` | Valid file path | `/models/generation_config.json` | Path to default sampling parameters (`temperature`, `top_p`, `top_k`, repetition penalties). |
| `vision` | `OnOff` | `on`, `off` | `on` | Enables multimodal vision encoder (`vision.faf` loaded from weights directory; accepts `data:` URL images). When `off`, image parts return HTTP 400 and vision kernels are not loaded. |
| `vision-preprocess-limit` | `Int` | `1` to `16` | `2` | Maximum concurrent images processed (decode, resize, patch tensor pack) across request threads. Protects against host RAM memory spikes. Requests exceeding the limit receive a retryable HTTP 503. |

---

### 2.2 Network & HTTP Server

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `host` | `Text` | IP address or hostname | `0.0.0.0` | Network interface address to bind and listen for incoming HTTP connections. |
| `port` | `Int` | `1` to `65535` | `5300` | TCP port for HTTP/REST and WebSocket endpoints. |
| `model-id` | `Text` | Any non-empty string | `qwen3.8-flash-next` | Model identifier string reported by `/v1/models` and completion responses. |
| `model-aliases` | `TextOrOff` | Comma-separated strings, `off` | `off` | Additional model aliases for `/v1/models` to ensure client compatibility (e.g. `gpt-4o,qwen-turbo`). |
| `api-key` | `TextOrOff` | Secret token string, `off` | `off` | Optional Bearer API key required for client requests. `off` disables authentication. |
| `request-read-timeout-s` | `Int` | `1` to `3600` | `30` | Maximum socket idle time (seconds) while request body is streaming before aborting with HTTP 408. |
| `log-level` | `Text` | `error`, `warn`, `info`, `debug` | `info` | Logging verbosity: `error`, `warn`, `info` (per-request summary and startup), or `debug` (full MTP/PLE/cache row traces). |

---

### 2.3 Slot Memory & Unified Memory Architecture (UMA)

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `total-slots` | `Int` | `1` to `8` | `4` | Number of concurrent agent sessions served simultaneously with complete hardware state isolation. |
| `capacity`<br>*(alias: `slot-capacity`)* | `Int` | `4096` to `1048576` | `262144` | Context capacity per slot in tokens (256K tokens per active slot). |
| `total-capacity` | `Int` | `0` to `16777216` | `1048576` | Global physical capacity ceiling across all slots (4 slots x 262,144 = 1,048,576 tokens). `0` = unconstrained by a global ceiling. |
| `kv-precommit` | `OnOff` | `on`, `off` | `on` | **Critical for Strix Halo.** Statically pre-commits 100% of flat KV-cache pages into physical DRAM at startup. Eliminates virtual memory overcommit, runtime allocator locks, and ROCm SVM page faults. |
| `kv-cache` | `Text` | `q8`, `bf16` | `q8` | KV cache quantization format: `q8` (saves 50% DRAM vs BF16 with negligible perplexity difference) or `bf16`. |
| `rope-yarn-factor` | `Number` | `1.0` to `8.0` | `1` | Context extension multiplier using YaRN: positions = 262,144 x factor. `1` uses trained RoPE without scaling. |
| `placement` | `Text` | `device`, `host-experts` | `device` | Weight physical residency: `device` (unified GPU DRAM) or `host-experts` (MoE stacks pinned in host RAM for undersized GPUs; slow, development only). |

---

### 2.4 Prefill Scheduling & Interleaving (Pareto Frontier)

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `chunk` | `Int` | `1` to `16384` | `4096` | Solo prefill quantum in tokens. Matches the RDNA 3.5 compute saturation knee (40 CUs / 20 WGPs), maximizing GEMM bandwidth. |
| `chunk-shared` | `Int` | `0` to `16384` | `1024` | Prefill quantum used when another slot is actively decoding. Prevents inter-token latency (ITL) freezes on active agent turns while maintaining high MoE expert saturation. `0` = always full chunk. |

---

### 2.5 Compute Precision

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `gdn-precision` | `Text` | `half-u`, `half`, `fast`, `exact` | `half-u` | Precision for GDN linear attention recurrent gates: `half-u` (FP16 inputs with FP32 accumulator via WMMA, U-path products as F16 hi + lo), `half` (one F16 product), `fast` (one BF16 product), or `exact` (BF16 hi + lo products). |
| `attention-precision` | `Text` | `fast`, `exact` | `fast` | Precision for QSA windowed attention softmax and dot-product kernels: `fast` (P as one BF16 term, faster execution) or `exact` (P as BF16 hi + lo products). |

---

### 2.6 Multi-Token Prediction (MTP) Speculative Drafting

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `mtp` | `OnOff` | `on`, `off` | `on` | Enables hardware speculative decoding via the decoupled MTP draft head. |
| `mtp-draft` | `Int` | `1` to `16` | `7` | Maximum speculative chain length per verification pass (up to 7 candidate tokens). |
| `mtp-margin` | `Number` | `0.0` to `100.0` | `1.5` | Base logit margin between top-1 and top-2 candidates under greedy decoding (T = 0). Candidate is drafted only if logit_1 - logit_2 >= margin. |
| `mtp-escalating-margin` | `OnOff` | `on`, `off` | `on` | Increases margin threshold with each successive step in the draft chain to suppress tail hallucinations. |
| `mtp-margin-slope` | `Number` | `0.0` to `1.0` | `0.08` | Margin increase per draft step: margin_k = margin_0 + k * slope. |
| `mtp-gate-rule` | `Text` | `margin`, `cost` | `margin` | Gating rule for greedy decoding (T = 0): `margin` (deterministic exactness) or `cost`. |
| `sampled-gate` | `Text` | `cost`, `margin` | `cost` | Gating rule for stochastic sampling (T > 0): `cost` (evaluates expected time speedup using empirical verify latency tables) or `margin`. |
| `mtp-vocab-filter` | `Text` | `all`, `en`, or `en+<blocks>` | `all` | Lossless draft-head vocabulary filter. Default `all` retains all 248,320 rows (slower drafting, emits startup warning). Filter mode requires mandatory `en` (core + English/code) with blocks combined via `+`: `eu`, `ru` (`cyrillic`), `zh` (`cn`), `ja` (`jp`), `ko` (`kr`), `ar` (`fa`), `other`. See Section 4. |
| `draft-head-q4` | `OnOff` | `on`, `off` | `on` | Quantizes draft head LM rows to Q4 at startup. Superseded when `draft-head-2stage = on`. |
| `draft-head-2stage` | `OnOff` | `on`, `off` | `on` | Two-Stage Act-PCA MTP draft head (`R=512`, `M_cand=512`, `cap=1024`, exact Q8 gather). **100% lossless** (`KLD = 0.0000`), cuts draft step latency ~30–44%, and saves ~126 MB VRAM vs. legacy Q4 draft (`en+ru`). See Section 5. |
| `lm-head-2stage` | `OnOff` | `on`, `off` | `on` | Two-Stage Act-PCA verify LM head (`R=512`, `M_cand=1536`, `cap=2048`, exact Q8 gather). Saves 2.2–2.7 ms per verify round (+6–12% end-to-end tok/s). Automatically falls back to Full Q8 on `top_k=0`, `top_k > 20`, grammars, `logit_bias`, and FP32. See Section 5. |
| `mtp-calib` | `OnOff` | `on`, `off` | `on` | Executes automated hardware latency calibration on physical GPU silicon at server startup. |
| `mtp-calib-cache` | `OnOff` | `on`, `off` | `on` | Persists calibrated profile to `mtp_calib_cache.json` alongside model weights for instant subsequent launches. |
| `mtp-calib-cache-file` | `PathOrOff` | Valid file path, `off` | `off` | Explicit path to calibration JSON file (`off` = automatic based on model directory). |
| `mtp-calib-draft-samples` | `Int` | `10` to `5000` | `300` | Number of timing samples for draft head latency calibration. |
| `mtp-calib-verify-samples` | `Int` | `10` to `500` | `50` | Number of timing samples for verify pass latency calibration. |
| `mtp-calib-warmup` | `Int` | `0` to `100` | `15` | Warmup iterations discarded before timing measurement. |
| `mtp-calib-tolerance-ms` | `Number` | `0.001` to `10.0` | `0.08` | Latency convergence tolerance in ms for early stopping. |
| `mtp-calib-triplets` | `Int` | `50` to `10000` | `1000` | Number of draft/verify triplets used for logistic regression. |
| `mtp-lut-step-penalty` | `Number` | `0.0` to `1.0` | `0.015` | MTP cost-gating empirical LUT step penalty deduction. |
| `mtp-ema-decay` | `Number` | `0.01` to `1.0` | `0.15` | MTP cost-gating Renewal-Reward exponential moving average (EMA) smoothing factor. |

---

### 2.7 Prefix Lookahead Engine (PLE)

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `ngram-cache-rows` | `Int` | `0` to `67108864` | `8388608` | Capacity of hot n-gram table LRU row cache in host RAM (8,388,608 rows ≈ 1.5 GiB RAM). Enables instant RAM-tier lookahead without NVMe disk latency. `0` disables row caching. |

---

### 2.8 Persistent Multi-Tier Prompt Cache

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `prompt-cache` | `OnOff` | `on`, `off` | `on` | Multi-tier prompt cache (RAM + NVMe). Eliminates redundant prefill for shared system instructions and prior multi-turn dialog context. |
| `prompt-cache-dir` | `Path` | Valid directory path | `/models/prompt-cache` | Directory path on NVMe storage for persistent session state snapshots. |
| `prompt-cache-gib` | `Int` | `0` to `4096` | `128` | Maximum disk storage allocation (in GiB) for cached prompt states. |
| `prompt-cache-ram-margin-gib`| `Int` | `0` to `64` | `4` | Host RAM safety floor (in GiB); when free RAM drops below this margin, idle snapshots are flushed from RAM to NVMe. |
| `low-memory` | `Text` | `adapt`, `fail` | `adapt` | Memory governor behavior: `adapt` (smoothly evicts RAM tier caches under pressure) or `fail` (refuses startup if RAM floor is violated). |
| `prompt-cache-idle-s` | `Int` | `1` to `86400` | `120` | Session idle timeout (seconds) before parking inactive slot states to NVMe. |
| `prompt-cache-write-gib-per-hour` | `Int` | `1` to `1024` | `48` | Maximum disk write bandwidth throttle (GiB/hour) to protect NVMe endurance. |
| `prompt-cache-disk-free-gib` | `Int` | `1` to `1024` | `32` | Minimum free disk space floor required before pausing snapshot persistence. |
| `prompt-cache-checkpoints` | `Int` | `1` to `64` | `16` | Maximum full session checkpoints retained per conversation. |
| `prompt-cache-points` | `Int` | `1` to `128` | `32` | Maximum incremental delta resume points preserved. |
| `prompt-cache-delta-max-tokens` | `Int` | `1024` to `1048576` | `65536` | Token delta ceiling before forcing a fresh full checkpoint snapshot. |

---

### 2.9 Reasoning & Tool-Calling Guards (Thinking Engine)

| Key | Type | Options | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `thinking` | `OnOff` | `on`, `off` | `on` | Enables reasoning output blocks (`<think>...</think>`) for requests that do not specify thinking parameters. |
| `thinking-cap` | `Int` | `0` to `1048576` | `0` | Upper bound on generated thinking tokens (`0` = unconstrained). Forces reasoning termination and triggers answer generation. |
| `thinking-cap-grace` | `Int` | `0` to `65536` | `256` | Token grace period past `thinking-cap` allowing the model to finish its active sentence before closing the block. `0` = hard stop at the cap. |
| `think-nudge` | `OnOff` | `on`, `off` | `on` | Loop detection heuristic. Detects repetitive reasoning patterns and nudges the model to commit to a decision. |
| `think-nudge-rate` | `Number` | `0.01` to `1.0` | `0.25` | Repetition threshold: triggers when 25% of the last 1024 thinking tokens repeat earlier 8-grams. |
| `think-nudge-min-tokens` | `Int` | `1024` to `16777216` | `3072` | Minimum thinking tokens generated before cycle detection activates. |
| `think-nudge-wording` | `Text` | `commit`, `decide` | `commit` | Guidance prompt wording: `commit` (*"... commit to the most promising approach now"*) or `decide` (*"... state my decision and the exact next command"*). |
| `think-end-guard` | `OnOff` | `on`, `off` | `on` | Prevents turn finalization if `<think>` or `<tool_call>` blocks remain open. |
| `think-close-on-call` | `OnOff` | `on`, `off` | `on` | Automatically injects `</think>` before an opened `<tool_call>` if the model initiates a call inside reasoning. |
| `literal-tags` | `OnOff` | `on`, `off` | `on` | Treats tag strings as literal text unless immediately preceded by a newline (guards against hallucinated formatting tags). |

---

## 3. Production Golden Configuration (`deploy/fastasfuck-server.conf`)

The following verified configuration represents the benchmarked production deployment for **AMD Strix Halo (128 GB LPDDR5X-8533, 40 CUs RDNA 3.5)**:

```ini
# ==============================================================================
# fastasfuck-server production configuration for AMD Strix Halo (gfx1151)
# 128 GB Unified Memory Architecture | 4 Concurrent Multi-Agent Slots
# ==============================================================================

# --- Model Artifacts ---
weights = /models/weights.faf
ngram = /models/ngram.table
tokenizer = /models/tokenizer.json
generation-config = /models/generation_config.json
model-id = qwen3.8-flash-next

# --- HTTP & Concurrency ---
host = 0.0.0.0
port = 5300
request-read-timeout-s = 30

# --- Slot Memory Controls (feat/slot-memory-controls) ---
total-slots = 4
capacity = 262144
total-capacity = 1048576
kv-precommit = on
kv-cache = q8
rope-yarn-factor = 1

# --- Prefill Scheduling (Pareto Saturation Knee) ---
chunk = 4096
chunk-shared = 1024

# --- Compute Precision ---
gdn-precision = half-u
attention-precision = fast

# --- MTP Speculative Drafting ---
mtp = on
mtp-draft = 7
mtp-margin = 1.5
mtp-escalating-margin = on
mtp-margin-slope = 0.08
mtp-gate-rule = margin
sampled-gate = cost
mtp-lut-step-penalty = 0.015
mtp-ema-decay = 0.15
mtp-vocab-filter = all
draft-head-q4 = on
draft-head-2stage = on
lm-head-2stage = on

# --- PLE n-gram Row Cache ---
ngram-cache-rows = 8388608

# --- Multi-Tier Prompt Cache ---
low-memory = adapt
prompt-cache = on
prompt-cache-dir = /models/prompt-cache
prompt-cache-gib = 128
prompt-cache-ram-margin-gib = 4
prompt-cache-idle-s = 120
prompt-cache-write-gib-per-hour = 48
prompt-cache-disk-free-gib = 32
prompt-cache-checkpoints = 16
prompt-cache-points = 32
prompt-cache-delta-max-tokens = 65536

# --- Reasoning & Guardrails ---
thinking = on
thinking-cap = 0
thinking-cap-grace = 256
think-nudge = on
think-end-guard = on
literal-tags = on
think-close-on-call = on
think-nudge-rate = 0.25
think-nudge-min-tokens = 3072
think-nudge-wording = commit

# --- Vision ---
vision = on
vision-preprocess-limit = 2
```

---

## 4. Draft Vocabulary Filtering (`--mtp-vocab-filter`)

`--mtp-vocab-filter` prunes unused language blocks from the MTP **draft** head matrix (`lm_head.mtp_draft_q4`) while keeping the full 248,320-token base model `lm_head` untouched.

> [!NOTE]
> The legacy `--mtp-vocab` integer flag has been removed. Passing `mtp-vocab` in a config file or CLI flag fails immediately at startup with a migration hint to use `--mtp-vocab-filter`.

### Default Behavior (`all`)
The built-in default is:
```ini
mtp-vocab-filter = all
```
This retains all 248,320 vocabulary rows (~357.6 MB in Q4), guaranteeing out-of-the-box support for every language. Because scanning the full 248k vocabulary on every draft step adds unnecessary LPDDR5X traffic, the server emits a startup **warning** when `all` is active so operators remember to pick their working languages.

### Language Blocks & Syntax (`+`-separated)
Whenever filtering is enabled (`!= all`), `core` + `en` are **mandatory** and cannot be omitted — Qwen 3.8 performs internal `<think>` reasoning in English, and stripping `core` special tokens (`<think>`, `</think>`, `<tool_call>`, `<|im_end|>`) or raw byte-fallback tokens breaks speculative drafting. Trying to run without `en` (e.g. `ru` or `eu` alone) fails fatally at startup.

Combine blocks with `+`:

| Block Tag (Aliases) | Included Vocabulary Content | Qwen 3.8 Rows | Q4 Footprint | Example Config |
| :--- | :--- | :---: | :---: | :--- |
| `all` | Full unfiltered vocabulary (emits startup warning) | 248,320 | ~357.6 MB | `mtp-vocab-filter = all` |
| `en` | **Mandatory base:** `core` (special tokens, 256 bytes, whitespace, digits, code/math symbols, single Greek math letters) + English/code ASCII-Latin (Epoch 1) | 93,294 | ~134.3 MB | `mtp-vocab-filter = en` |
| `+eu` | European Latin: multilingual ASCII-Latin words (auto-detected post-Han Epoch 3, `eu_ascii_start=151580`) + European Latin-1 / Latin-Ext diacritics (`de`, `fr`, `es`, `it`, `pt`, `pl`, `cs`, `hr`, `tr`, `nordic`, ...) | +47,272 | +68.1 MB | `mtp-vocab-filter = en+eu` (140,566 rows, 202.4 MB) |
| `+ru` (`cyrillic`) | Entire Cyrillic Unicode block (Russian, Ukrainian, Belarusian, Bulgarian, Serbian, etc.) + Cyrillic partial UTF-8 BPE fragments | +18,582 | +26.8 MB | `mtp-vocab-filter = en+ru` (111,876 rows, 161.1 MB) |
| `+zh` (`cn`) | Han / CJK ideographs + CJK punctuation/fullwidth + Han UTF-8 fragments | +53,978 | +77.7 MB | `mtp-vocab-filter = en+zh` (147,272 rows, 212.1 MB) |
| `+ja` (`jp`) | Hiragana + Katakana + CJK punctuation/fullwidth (automatically includes `zh` Kanji) | +59,457 | +85.6 MB | `mtp-vocab-filter = en+ja` (152,751 rows, 220.0 MB) |
| `+ko` (`kr`) | Hangul Syllables/Jamo + CJK punctuation/fullwidth + Hangul UTF-8 fragments | +7,277 | +10.5 MB | `mtp-vocab-filter = en+ko` (100,571 rows, 144.8 MB) |
| `+ar` (`fa`) | Arabic & Persian (`U+0600..0x06FF`, `U+FE70..0xFEFE`, excluding `U+FEFF` BOM) + Arabic UTF-8 fragments | +8,821 | +12.7 MB | `mtp-vocab-filter = en+ar` (102,115 rows, 147.0 MB) |
| `+other` | Vietnamese (`U+1EA0..0x1EF9`, `đ/ơ/ư`), Thai, Indic scripts, Greek words, Hebrew, rare scripts & Emoji | +13,524 | +19.5 MB | `mtp-vocab-filter = en+other` (106,818 rows, 153.8 MB) |

Common multi-language combinations:
- **English + Code + Russian:** `mtp-vocab-filter = en+ru` (`111,876` rows, `161.1 MB` Q4)
- **English + Code + European + Russian:** `mtp-vocab-filter = en+eu+ru` (`159,148` rows, `229.2 MB` Q4)
- **English + Code + Japanese (Kana + Kanji):** `mtp-vocab-filter = en+ja` (`152,751` rows, `220.0 MB` Q4)

### Measured Speedup & Determinism (`en+ru` Lossless, Strix Halo `gfx1151`)
Switching from the legacy `en+ru` filter (160,904 rows / 231.7 MB, which included Epoch-3 European Latin) to the separated lossless `en+ru` filter (`111,876` rows / `161.1 MB`, **−30.5%** draft-head footprint):
- **Draft Step Calibration Latency:** drops from **1.58 ms → 1.28 ms** (−19.0%).
- **Greedy Decoding (`T = 0.0`, Solo P1–P6, n=5):** **71.12 → 72.69 tok/s** (+2.2%; P2 hits **90.29 tok/s**) with **100.000% SHA-256 bit-identity** (`d4e53aafaf10e70f`, `9f51ac8d73f49616`, `d2e35593c5e11eb1`, `3754e442b5144c4e`, `59a66b4cf7b69c29`, `ffc67dbf037d2790`), because the full-precision base verifier always evaluates against the complete 248,320-token vocabulary.
- **Sampling Performance (`T = 0.6` / `T = 1.0`, `--prompt-cache off`):**
  - `T = 0.6` Solo (`P1/P2/P4/P6`): **75.14 → 76.79 tok/s**
  - `T = 0.6` 4-Slot Concurrent: **114.95 → 116.96 ± 2.48 tok/s** (`per_slot 33.78 tok/s`)
  - `T = 1.0` Solo (`P1/P4/P6`): **70.03 → 73.09 tok/s**
  - `T = 1.0` 33k `<think>`: **57.94 → 61.25 tok/s** (`+5.7%`, acceptance **69.32%**)
- **Automatic Calibration Isolation:** `mtp_calib_cache.json` fingerprints automatically hash the retained compact token ID list (`canonical#rows#hash64`) without invalidating `.pcache` KV snapshots.

---

## 5. Two-Stage Act-PCA Vocabulary Heads (`--draft-head-2stage` & `--lm-head-2stage`)

On AMD Strix Halo (`32 MiB` MALL cache, `273 GB/s` LPDDR5X), scanning the dense `lm_head.weight` (`~676 MB` in Q8 across 248,320 rows) or legacy dense Q4 draft heads (`161–358 MB`) thrashes the MALL cache on every decode and verify step. Beginning in **Release 1.3.0**, **both** two-stage Activation-PCA heads are enabled out of the box:

```ini
draft-head-2stage = on
lm-head-2stage = on
```

### 5.1 Three-Stage GPU Pipeline & Startup Caching
1. **First-Startup Calibration (`draft2s_pca_r512.bin`, ~2.6 MB):** Once per model weight + calibration fingerprint (`SPCA v2`), the server captures `1,022` real `final_mixed` BF16 hidden states covering code, math, Linux internals, Russian prose, `<think>` reasoning, JSON tool calls, CJK, and rare Unicode symbols (`runtime/act_pca.cpp`), solving an exact FP64 dual-Gram eigensystem (`U U^T`) for orthonormal rank `R = 512` directions `E [512, K]` in ~5 s. Ensure the weights directory is writable on first startup so `draft2s_pca_r512.bin` can be cached next to `weights.faf` (restarts then load it in `~0.2 s`; read-only directories recompute cleanly in ~5 s). `mtp_calib_cache.json` automatically isolates calibration profiles via `+2s_r512_m512` and `+lm2s_m1536` tags.
2. **Stage 1 & 2 (`kernels/draft2s.hip`):** Projects hidden states `x_r = x * E^T` (`2560 -> 512` BF16), scores compressed Q4 (`G = 64`) projection weights `P_q4`, and shortlists the top `M_cand` rows (`<= cap`) via a zero-overflow 2-pass 24-bit radix histogram:
   - **`draft-head-2stage` (`M_cand = 512, cap = 1024`):** `35.3 MB` (`en+ru`) or `74.1 MB` (`all`) — saves ~`126 MB` vs. legacy Q4 `en+ru` (`161.1 MB`), cuts draft latency by ~30–44% (`1.28 -> 0.88–0.89 ms`), and lifts draft Top-1 match vs. Q8 from `96.5%` to `100.0%`.
   - **`lm-head-2stage` (`M_cand = 1536, cap = 2048`):** `74.1 MB` over the full 248,077-token vocabulary (vs. `676 MB` Full Q8), saving `2.2–2.7 ms` per verify round (+6–12% end-to-end tok/s). Together, `draft2s + lm2s` (`109.4 MB` on `en+ru`) fit in less memory than the legacy 1.2.2 Q4 draft head alone.
3. **Stage 3 (`q8_gather_rows_kernel`):** Reads and dot-products the exact unmodified Q8 `lm_head.weight` rows for only the `<= cap` shortlisted candidates, then extracts top-20 candidates / `MtpPick`.

### 5.2 Safety, Parity & Automatic Fallback
- **`--draft-head-2stage` (`on`):** Strictly **100% lossless (`KLD = 0.0000`)** under both greedy (`T = 0`) and Leviathan (`T > 0`) decoding — the base verifier checks every proposed token against full-precision verify logits and cleanly rejects/resamples any miss with zero distribution bias.
- **`--lm-head-2stage` (`on`):** Verified against Full Q8 across live generation and OOD stress suites:
  - **Greedy (`T = 0.0`, P1–P6):** **6/6 (100.000%) exact SHA-256 bit-identity** with Full Q8 (`76.49 -> 80.36 tok/s`).
  - **Live decode (`T = 0.6`):** **0/936** nucleus misses across `<think>` decode, real response decode, and EN/RU prompts (`KLD ~ 3e-13`). **0/128** misses on `<think>` at `T = 1.0` (`KLD ~ 1.4e-13`).
  - **Automatic Full-Q8 Fallback:** `lm_head_2s` runs only when `Sampler::takes_candidates()` is true (`T = 0` or `top_k` in `1..20`), activations are `BF16`, `mask_rows == 0`, and `bias_n == 0`. Requests with `top_k = 0`, `top_k > 20`, structured output / JSON grammars, `logit_bias`, or FP32 automatically execute Full Q8 `lm_head`.

### 5.3 Why `~3e-13` KLD Appears vs. Full Q8 (Not a Math Bug or WMMA)
Live decode audits show `TVD_mean ≈ 9.2e-8 .. 1.14e-7` and `KLD_mean ≈ 1.4e-13 .. 3.04e-13` between `lm-head-2stage` and Full Q8. This `~3e-13` residual is the hardware FP32 reduction-order floor — **not** quantization inaccuracy and **not** WMMA (FP16/BF16 WMMA rounding would produce `~1e-3..1e-2` TVD and `~1e-6` KLD, a million times larger):
1. **Exact Stage-3 Q8 Math:** `q8_gather_rows_kernel` (`kernels/draft2s.hip`) executes the exact same asymmetric Q8 `v_dot2_f32_bf16` (`__builtin_amdgcn_fdot2_f32_bf16`) arithmetic on the exact same `lm_head.weight` bytes as `linear_q8_kernel` (`kernels/linear_q8.hip`).
2. **1-ULP FP32 Associativity Difference:** Because FP32 FMA accumulation is non-associative (`(a + b) + c != a + (b + c)`), the slightly different warp/thread parallel reduction tree shifts logits by at most **~1 ULP in FP32** (`ε_FP32 = 2^-23 ≈ 1.19e-7`), giving `TVD ~ 1e-7`.
3. **Quadratic KLD Scaling Near `q = p`:** Expanding KL divergence around `q = p` yields `KLD(p || q) ≈ (1/2) * sum_i ((δp_i)^2 / p_i) ~ (ε_FP32)^2 ~ 1e-14 .. 3e-13`. Dividing that same 1-ULP logit delta by `T = 0.6` instead of `T = 1.0` scales `(δp)^2` by `(1/0.6)^2 ≈ 2.78x`, explaining `1.64e-13 -> 3.04e-13`.

### 5.4 When to Disable (`off`)
- **`--lm-head-2stage off`:** Disable if you run high-temperature (`T >= 1.0`) creative writing in CJK or rare scripts with `top_k` in `1..20` and want zero tail pruning, run strict full-vocab perplexity/KLD reference evals against Full Q8, or send exclusively `top_k = 0` / grammar-constrained traffic (already handled via auto-fallback) and want to reclaim `74.1 MB` VRAM.
- **`--draft-head-2stage off`:** Rarely needed — keep `on` unless A/B-benchmarking against the legacy Q4 draft head.
