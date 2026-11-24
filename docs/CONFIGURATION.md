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
| `mtp-vocab` | `Int` | `0` to `1073741824` | `0` | Restricts draft head scoring to token range [0, N). `0` scores over the full vocabulary. |
| `mtp-vocab-filter` | `Text` | `all`, `en+ru`, `en`, `no-cjk`, or section list | `all` | Prunes draft head matrix rows by Unicode script. Presets: `all` (or `full`), `en+ru` (or `ru+en`), `en` (or `coding`), `no-cjk` (or `western`), or comma-separated sections (`core,latin,cyrillic,greek,digits,punct,code,cjk,arabic,asian,other`) where `core` is mandatory. |
| `draft-head-q4` | `OnOff` | `on`, `off` | `on` | Quantizes draft head LM rows to Q4 at startup. Reduces draft step latency from 2.11 ms to 1.57 ms while keeping the base verifier in full precision. |
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
mtp-vocab = 0
mtp-vocab-filter = all
draft-head-q4 = on

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

## 4. Vocabulary Filtering: Universal `all` vs Localized `en+ru`

The default configuration defines:
```ini
mtp-vocab-filter = all
```
This retains all 248,320 vocabulary tokens in the draft head matrix (357.6 MB in Q4), providing out-of-the-box multilingual support across all scripts, Asian languages, mathematical symbols, and code.

For environments focused on English, Russian, and programming workloads, overriding this parameter delivers tangible speedups:
```ini
mtp-vocab-filter = en+ru
```
- **Mechanism:** Prunes unused non-Latin / non-Cyrillic rows from the draft head matrix, keeping 160,904 tokens (231.7 MB, -35.2% matrix footprint).
- **Draft Latency:** Reduces draft step latency from 2.11 ms down to **1.57 ms** (-25.6%).
- **Throughput:** Delivers a sustained **+4.4% to +4.6% decode throughput increase** across all prompts (70.64 -> 73.72 tok/s on Thinking-Cap).
- **Bit-Identity Guarantee:** Output sequences under greedy decoding (T = 0) are **100.000% bit-identical** (`H_all == H_en_ru`) because the full-precision base verifier checks all candidates against the complete base vocabulary.
