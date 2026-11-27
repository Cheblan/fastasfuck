#include "serve/vocab_filter.hpp"

#include "common/check.hpp"
#include "serve/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace strix::vocab_filter {

namespace {

enum class CharClass {
    NeutralCore,
    AsciiLatin,
    EuExt,
    Cyrillic,
    Han,
    Kana,
    Hangul,
    CjkPunct,
    Arabic,
    GreekLetter,
    Other,
};

std::string trim(const std::string &s) {
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string to_lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool decode_utf8_safe(std::string_view s, std::vector<uint32_t> &out) {
    out.clear();
    const auto *p = reinterpret_cast<const uint8_t *>(s.data());
    const size_t len = s.size();
    size_t i = 0;
    while (i < len) {
        const uint8_t c = p[i];
        if (c < 0x80) {
            out.push_back(c);
            ++i;
        } else if ((c >> 5) == 0x06) {
            if (i + 1 >= len || (p[i + 1] >> 6) != 0x02) return false;
            const uint32_t cp = ((c & 0x1F) << 6) | (p[i + 1] & 0x3F);
            if (cp < 0x80) return false;
            out.push_back(cp);
            i += 2;
        } else if ((c >> 4) == 0x0E) {
            if (i + 2 >= len || (p[i + 1] >> 6) != 0x02 || (p[i + 2] >> 6) != 0x02) return false;
            const uint32_t cp = ((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
            out.push_back(cp);
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            if (i + 3 >= len || (p[i + 1] >> 6) != 0x02 || (p[i + 2] >> 6) != 0x02 || (p[i + 3] >> 6) != 0x02) {
                return false;
            }
            const uint32_t cp =
                ((c & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) return false;
            out.push_back(cp);
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

bool is_greek_math_letter(uint32_t cp) {
    // Standard single-letter lowercase α..ω (0x03B1..0x03C9) and common uppercase physics/math Greek symbols.
    if (cp >= 0x03B1 && cp <= 0x03C9) return true;
    switch (cp) {
        case 0x0393:  // Γ
        case 0x0394:  // Δ
        case 0x0398:  // Θ
        case 0x039B:  // Λ
        case 0x039E:  // Ξ
        case 0x03A0:  // Π
        case 0x03A3:  // Σ
        case 0x03A6:  // Φ
        case 0x03A8:  // Ψ
        case 0x03A9:  // Ω
            return true;
        default:
            return false;
    }
}

bool is_vietnamese_specific(uint32_t cp) {
    if (cp >= 0x1EA0 && cp <= 0x1EF9) return true;
    return cp == 0x0110 || cp == 0x0111 ||  // Đ, đ
           cp == 0x01A0 || cp == 0x01A1 ||  // Ơ, ơ
           cp == 0x01AF || cp == 0x01B0;    // Ư, ư
}

CharClass classify_codepoint(uint32_t cp) {
    // Whitespace & control, including U+FEFF BOM (specifically excluded from Arabic FE70..FEFE)
    if (cp < 0x20 || cp == 0x20 || cp == 0x7F || cp == 0xFEFF) return CharClass::NeutralCore;

    // ASCII digits & ASCII punctuation/code operators
    if (cp >= 0x30 && cp <= 0x39) return CharClass::NeutralCore;
    if ((cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
        (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E)) {
        return CharClass::NeutralCore;
    }

    // ASCII Latin [A-Za-z]
    if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A)) return CharClass::AsciiLatin;

    // Latin-1 symbols/punct + General Punctuation + Currency + Letterlike/Math/Technical/Box/Shapes
    if ((cp >= 0x00A0 && cp <= 0x00BF) || (cp >= 0x2000 && cp <= 0x20CF) || (cp >= 0x2100 && cp <= 0x2BFF)) {
        return CharClass::NeutralCore;
    }
    // Combining marks / spacing modifiers
    if (cp >= 0x02B0 && cp <= 0x036F) return CharClass::NeutralCore;

    // Vietnamese-specific Latin diacritics go to Other
    if (is_vietnamese_specific(cp)) return CharClass::Other;

    // Non-ASCII Latin (European diacritics + IPA)
    if ((cp >= 0x00C0 && cp <= 0x02AF) || (cp >= 0x1E00 && cp <= 0x1EFF) ||
        (cp >= 0x2C60 && cp <= 0x2C7F) || (cp >= 0xA720 && cp <= 0xA7FF)) {
        return CharClass::EuExt;
    }

    // Greek
    if ((cp >= 0x0370 && cp <= 0x03FF) || (cp >= 0x1F00 && cp <= 0x1FFF)) return CharClass::GreekLetter;

    // Cyrillic
    if ((cp >= 0x0400 && cp <= 0x052F) || (cp >= 0x2DE0 && cp <= 0x2DFF) || (cp >= 0xA640 && cp <= 0xA69F)) {
        return CharClass::Cyrillic;
    }

    // Arabic & Persian (explicitly capped at 0xFEFE so U+FEFF BOM stays NeutralCore)
    if ((cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) || (cp >= 0x08A0 && cp <= 0x08FF) ||
        (cp >= 0xFB50 && cp <= 0xFDFF) || (cp >= 0xFE70 && cp <= 0xFEFE)) {
        return CharClass::Arabic;
    }

    // Japanese Kana (Hiragana + Katakana + Halfwidth Katakana)
    if ((cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF) ||
        (cp >= 0x31F0 && cp <= 0x31FF) || (cp >= 0xFF65 && cp <= 0xFF9F)) {
        return CharClass::Kana;
    }

    // Korean Hangul
    if ((cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F)) {
        return CharClass::Hangul;
    }

    // Han / CJK Ideographs
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x3100 && cp <= 0x312F) || (cp >= 0x20000 && cp <= 0x2CEAF)) {
        return CharClass::Han;
    }

    // CJK punctuation / fullwidth forms
    if ((cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFF64) ||
        (cp >= 0xFFA0 && cp <= 0xFFEF) || (cp >= 0xFE30 && cp <= 0xFE6F)) {
        return CharClass::CjkPunct;
    }

    return CharClass::Other;
}

bool is_pure_ascii_latin_token(const std::string &raw) {
    if (raw.empty()) return false;
    bool has_alpha = false;
    for (unsigned char c : raw) {
        if (c >= 0x80) return false;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) has_alpha = true;
    }
    return has_alpha;
}

// Detect Epoch-3 multilingual ASCII-Latin start ID without hardcoded constants.
// Finds the first >= 8 Ki-token contiguous run of 1024-ID blocks with sparse ASCII-Latin (< 50/block),
// such as Qwen's [95k..148k) Han block, and returns the first pure ASCII-Latin ID after that wall ends.
int64_t detect_eu_ascii_start(const Tokenizer &tok, int64_t max_vocab) {
    constexpr int64_t kBlock = 1024;
    constexpr int kLowBlocksRequired = 8;
    constexpr int kLowAsciiThresh = 50;

    const int64_t n_blocks = (max_vocab + kBlock - 1) / kBlock;
    if (n_blocks <= kLowBlocksRequired + 1) return max_vocab;

    int low_run = 0;
    int64_t wall_end_block = -1;
    // Skip block 0 (single-byte & core tokens [0..1024)).
    for (int64_t b = 1; b < n_blocks; ++b) {
        const int64_t lo = b * kBlock;
        const int64_t hi = std::min(max_vocab, lo + kBlock);
        int ascii_cnt = 0;
        for (int64_t id = lo; id < hi; ++id) {
            if (!tok.is_added(static_cast<int32_t>(id)) &&
                is_pure_ascii_latin_token(tok.token_bytes(static_cast<int32_t>(id)))) {
                ++ascii_cnt;
            }
        }
        if (ascii_cnt < kLowAsciiThresh) {
            ++low_run;
        } else {
            if (low_run >= kLowBlocksRequired) {
                wall_end_block = b;
                break;
            }
            low_run = 0;
        }
    }
    if (wall_end_block < 0) return max_vocab;

    const int64_t scan_lo = std::max<int64_t>(0, (wall_end_block - 1) * kBlock);
    for (int64_t id = scan_lo; id < max_vocab; ++id) {
        if (!tok.is_added(static_cast<int32_t>(id)) &&
            is_pure_ascii_latin_token(tok.token_bytes(static_cast<int32_t>(id)))) {
            return id;
        }
    }
    return max_vocab;
}

bool allow_partial_utf8(const std::string &raw, uint32_t langs) {
    uint8_t lead = 0;
    for (unsigned char c : raw) {
        if (c >= 0x80) {
            lead = c;
            break;
        }
    }
    if (lead == 0) return (langs & Lang::Core) != 0;
    if (0xD0 <= lead && lead <= 0xD3) return (langs & Lang::Ru) != 0;
    if (0xD8 <= lead && lead <= 0xDB) return (langs & Lang::Ar) != 0;
    if (0xC3 <= lead && lead <= 0xCD) return (langs & Lang::Eu) != 0;
    if ((0xCE <= lead && lead <= 0xCF) || (0xD4 <= lead && lead <= 0xD7) || (0xE0 <= lead && lead <= 0xE1)) {
        return (langs & Lang::Other) != 0;
    }
    if (lead == 0xE3) return (langs & (Lang::Ja | Lang::Zh | Lang::Ko)) != 0;
    if (0xE4 <= lead && lead <= 0xE9) return (langs & (Lang::Zh | Lang::Ja)) != 0;
    if (0xEA <= lead && lead <= 0xED) return (langs & Lang::Ko) != 0;
    // C2 (Unicode symbols/NBSP), E2 (general punct/math), EF (fullwidth/BOM), F0..F4 (emoji/ext), 80..BF (cont byte)
    return (langs & Lang::Core) != 0;
}

std::string canonical_spec_for(uint32_t langs) {
    std::vector<const char *> parts;
    if (langs & Lang::En) parts.push_back("en");
    if (langs & Lang::Eu) parts.push_back("eu");
    if (langs & Lang::Ru) parts.push_back("ru");
    if (langs & Lang::Zh) parts.push_back("zh");
    if (langs & Lang::Ja) parts.push_back("ja");
    if (langs & Lang::Ko) parts.push_back("ko");
    if (langs & Lang::Ar) parts.push_back("ar");
    if (langs & Lang::Other) parts.push_back("other");
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += '+';
        out += parts[i];
    }
    return out;
}

}  // namespace

FilterSpec parse_filter(const std::string &spec_raw) {
    const std::string trimmed = trim(spec_raw);
    const std::string lower = to_lower(trimmed);
    if (lower.empty() || lower == "all") {
        return FilterSpec{true, 0, "all"};
    }

    STRIX_CHECK(trimmed.find(',') == std::string::npos,
                "vocab_filter::parse_filter: use '+' instead of ',' in --mtp-vocab-filter '", spec_raw,
                "' (example: en+ru+ja+zh)");
    STRIX_CHECK(trimmed.front() != '+' && trimmed.back() != '+',
                "vocab_filter::parse_filter: empty tag in '", spec_raw, "'");

    uint32_t langs = 0;
    bool saw_all = false;
    std::stringstream ss(lower);
    std::string item;
    while (std::getline(ss, item, '+')) {
        const std::string t = trim(item);
        STRIX_CHECK(!t.empty(), "vocab_filter::parse_filter: empty tag in '", spec_raw, "'");
        if (t == "all") {
            saw_all = true;
        } else if (t == "core") {
            langs |= Lang::Core;
        } else if (t == "en") {
            langs |= Lang::Core | Lang::En;
        } else if (t == "eu") {
            langs |= Lang::Eu;
        } else if (t == "ru" || t == "cyrillic") {
            langs |= Lang::Ru;
        } else if (t == "zh" || t == "cn") {
            langs |= Lang::Zh;
        } else if (t == "ja" || t == "jp") {
            langs |= Lang::Ja | Lang::Zh;
        } else if (t == "ko" || t == "kr") {
            langs |= Lang::Ko;
        } else if (t == "ar" || t == "fa") {
            langs |= Lang::Ar;
        } else if (t == "other") {
            langs |= Lang::Other;
        } else {
            STRIX_FAIL("vocab_filter::parse_filter: unknown filter tag '", t,
                       "'; expected 'all' or '+'-separated tags from {en, eu, ru (cyrillic), zh (cn), ja (jp), "
                       "ko (kr), ar (fa), other}");
        }
    }

    STRIX_CHECK(!saw_all, "vocab_filter::parse_filter: 'all' cannot be combined with other tags in '", spec_raw, "'");

    if (!(langs & Lang::Core) || !(langs & Lang::En)) {
        std::fprintf(stderr,
                     "FATAL: --mtp-vocab-filter '%s' missing mandatory 'core'/'en' (special tokens, byte fallback, "
                     "and English thinking tokens are required). Use 'all', 'en', or 'en+<lang>'.\n",
                     spec_raw.c_str());
        STRIX_FAIL("vocab_filter::parse_filter: --mtp-vocab-filter '", spec_raw,
                   "' must include 'en' (mandatory core + English/code base); use 'all', 'en', or 'en+<lang>'");
    }

    return FilterSpec{false, langs, canonical_spec_for(langs)};
}

std::vector<int32_t> filter_vocabulary(const Tokenizer &tok, const FilterSpec &spec, int64_t max_vocab,
                                       int64_t *out_eu_ascii_start) {
    if (out_eu_ascii_start) *out_eu_ascii_start = max_vocab;
    if (spec.all) {
        return {};
    }

    if (!(spec.langs & Lang::Core) || !(spec.langs & Lang::En)) {
        std::fprintf(stderr, "FATAL: --mtp-vocab-filter must include 'core' and 'en'\n");
        std::abort();
    }

    const int64_t limit = std::min<int64_t>(max_vocab, tok.size());
    const int64_t eu_ascii_start = detect_eu_ascii_start(tok, limit);
    if (out_eu_ascii_start) *out_eu_ascii_start = eu_ascii_start;

    std::vector<int32_t> compact_to_vocab;
    compact_to_vocab.reserve(static_cast<size_t>(limit));

    std::vector<uint32_t> cps;
    for (int32_t id = 0; id < static_cast<int32_t>(limit); ++id) {
        const std::string &raw = tok.token_bytes(id);

        // 1. Added special tokens (<think>, </think>, <tool_call>, <|im_end|>, etc.) and single bytes -> always Core
        if (tok.is_added(id) || raw.size() <= 1) {
            compact_to_vocab.push_back(id);
            continue;
        }

        // 2. Partial / incomplete BPE UTF-8 fragments -> classify by first non-ASCII byte
        if (!decode_utf8_safe(raw, cps)) {
            if (allow_partial_utf8(raw, spec.langs)) {
                compact_to_vocab.push_back(id);
            }
            continue;
        }

        bool has_ascii_lat = false;
        bool has_non_neutral_non_ascii_lat = false;
        bool allowed = true;
        int greek_cnt = 0;
        uint32_t single_greek_cp = 0;

        for (uint32_t cp : cps) {
            const CharClass cc = classify_codepoint(cp);
            if (cc == CharClass::NeutralCore) continue;
            if (cc == CharClass::AsciiLatin) {
                has_ascii_lat = true;
                continue;
            }
            has_non_neutral_non_ascii_lat = true;
            switch (cc) {
                case CharClass::EuExt:
                    if (!(spec.langs & Lang::Eu)) allowed = false;
                    break;
                case CharClass::Cyrillic:
                    if (!(spec.langs & Lang::Ru)) allowed = false;
                    break;
                case CharClass::Han:
                    if (!(spec.langs & (Lang::Zh | Lang::Ja))) allowed = false;
                    break;
                case CharClass::Kana:
                    if (!(spec.langs & Lang::Ja)) allowed = false;
                    break;
                case CharClass::Hangul:
                    if (!(spec.langs & Lang::Ko)) allowed = false;
                    break;
                case CharClass::CjkPunct:
                    if (!(spec.langs & (Lang::Zh | Lang::Ja | Lang::Ko))) allowed = false;
                    break;
                case CharClass::Arabic:
                    if (!(spec.langs & Lang::Ar)) allowed = false;
                    break;
                case CharClass::GreekLetter:
                    ++greek_cnt;
                    single_greek_cp = cp;
                    if (!(spec.langs & Lang::Other)) allowed = false;
                    break;
                case CharClass::Other:
                    if (!(spec.langs & Lang::Other)) allowed = false;
                    break;
                case CharClass::NeutralCore:
                case CharClass::AsciiLatin:
                    break;
            }
            if (!allowed && cc != CharClass::GreekLetter) break;
        }

        // Single Greek math symbols (e.g. 'π', ' Δ', 'α') belong to Core math vocabulary.
        if (!allowed && greek_cnt == 1 && !has_ascii_lat && is_greek_math_letter(single_greek_cp)) {
            bool all_neutral_except_greek = true;
            for (uint32_t cp : cps) {
                const CharClass cc = classify_codepoint(cp);
                if (cc != CharClass::NeutralCore && cc != CharClass::GreekLetter) {
                    all_neutral_except_greek = false;
                    break;
                }
            }
            if (all_neutral_except_greek) allowed = true;
        }

        if (!allowed) continue;

        // Pure ASCII-Latin tokens after the auto-detected multilingual wall belong to Eu (unless paired with a selected non-Latin script).
        if (has_ascii_lat && !has_non_neutral_non_ascii_lat && static_cast<int64_t>(id) >= eu_ascii_start) {
            if (!(spec.langs & Lang::Eu)) continue;
        }

        compact_to_vocab.push_back(id);
    }

    return compact_to_vocab;
}

std::string cache_key(const FilterSpec &spec, const std::vector<int32_t> &ids) {
    if (spec.all) return "all";
    uint64_t h = 14695981039346656037ULL;
    for (int32_t id : ids) {
        uint64_t v = static_cast<uint32_t>(id);
        h ^= v;
        h *= 1099511628211ULL;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "#%zu#%016llx", ids.size(), static_cast<unsigned long long>(h));
    return spec.canonical + buf;
}

}  // namespace strix::vocab_filter
