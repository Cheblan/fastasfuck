#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

class Tokenizer;

namespace vocab_filter {

enum Lang : uint32_t {
    Core  = 1u << 0,  // Mandatory: added special tokens, 256 raw bytes, whitespace, digits, ASCII/Unicode punct/code, single Greek math letters
    En    = 1u << 1,  // Mandatory base: English/code ASCII-Latin (Epoch 1)
    Eu    = 1u << 2,  // European Latin: Epoch-3 multilingual ASCII-Latin + European diacritics
    Ru    = 1u << 3,  // Cyrillic block (~18.6k)
    Zh    = 1u << 4,  // Han ideographs + CJK punct/fullwidth
    Ja    = 1u << 5,  // Kana (Hiragana/Katakana) + CJK punct/fullwidth (tag ja/jp also enables Zh)
    Ko    = 1u << 6,  // Hangul Syllables/Jamo + CJK punct/fullwidth
    Ar    = 1u << 7,  // Arabic & Persian (excluding U+FEFF BOM)
    Other = 1u << 8,  // Vietnamese, Thai, Indic, Greek non-math, Hebrew, rare scripts & Emoji
};

struct FilterSpec {
    bool all = false;
    uint32_t langs = 0;
    std::string canonical = "all";
};

// Parse '--mtp-vocab-filter' specification string:
//   - "all" (or "full"): full unfiltered vocabulary
//   - "en" or '+'-separated tags combined with 'en':
//       en, eu, ru (alias cyrillic), zh (alias cn), ja (alias jp), ko (alias kr), ar (alias fa), other
// Throws via STRIX_FAIL on unknown tags, non-'+' delimiters (e.g. ','), missing mandatory 'en'/'core', or mixing with 'all'.
FilterSpec parse_filter(const std::string &spec_raw);

inline bool is_all(const FilterSpec &spec) {
    return spec.all;
}

// Retain sorted original token IDs in [0, max_vocab) matching `spec`.
// Crashes immediately (abort) if called on a non-'all' spec missing Core or En.
// If out_eu_ascii_start != nullptr, writes the auto-detected multilingual Epoch-3 ASCII-Latin start ID.
std::vector<int32_t> filter_vocabulary(const Tokenizer &tok, const FilterSpec &spec, int64_t max_vocab,
                                       int64_t *out_eu_ascii_start = nullptr);

// Deterministic calibration cache key segment combining canonical spec, row count, and 64-bit hash of retained IDs.
std::string cache_key(const FilterSpec &spec, const std::vector<int32_t> &ids);

}  // namespace vocab_filter
}  // namespace strix
