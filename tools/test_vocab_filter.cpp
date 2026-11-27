#include "common/check.hpp"
#include "serve/server_config.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vocab_filter.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using namespace strix;
using strix::vocab_filter::FilterSpec;
using strix::vocab_filter::Lang;

template <typename Fn>
bool throws_containing(Fn &&fn, const std::string &needle) {
    try {
        fn();
    } catch (const std::exception &e) {
        if (std::string(e.what()).find(needle) != std::string::npos) return true;
        std::fprintf(stderr, "Unexpected exception message: '%s' (expected substr '%s')\n", e.what(), needle.c_str());
        return false;
    }
    std::fprintf(stderr, "Expected throw containing '%s', but succeeded\n", needle.c_str());
    return false;
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

std::string cp_to_utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

std::string bytes_to_gpt2_text(const std::string &raw) {
    static std::vector<uint32_t> b2c = [] {
        std::vector<uint32_t> m(256);
        uint32_t next = 256;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            m[b] = printable ? b : next++;
        }
        return m;
    }();
    std::string out;
    for (unsigned char b : raw) out += cp_to_utf8(b2c[b]);
    return out;
}

void test_parse_filter() {
    {
        const FilterSpec s = vocab_filter::parse_filter("all");
        STRIX_CHECK(s.all && vocab_filter::is_all(s) && s.canonical == "all", "parse 'all' failed");
    }
    {
        const FilterSpec s = vocab_filter::parse_filter("  ALL ");
        STRIX_CHECK(s.all && s.canonical == "all", "parse '  ALL ' failed");
    }
    {
        const FilterSpec s = vocab_filter::parse_filter("en");
        STRIX_CHECK(!s.all && s.langs == (Lang::Core | Lang::En) && s.canonical == "en", "parse 'en' failed");
    }
    {
        const FilterSpec s1 = vocab_filter::parse_filter("en+ru");
        const FilterSpec s2 = vocab_filter::parse_filter(" en + cyrillic ");
        STRIX_CHECK(!s1.all && s1.langs == (Lang::Core | Lang::En | Lang::Ru) && s1.canonical == "en+ru",
                    "en+ru canonical mismatch: ", s1.canonical);
        STRIX_CHECK(s1.langs == s2.langs && s2.canonical == "en+ru", "cyrillic alias failed");
    }
    {
        const FilterSpec s1 = vocab_filter::parse_filter("en+eu");
        STRIX_CHECK(s1.langs == (Lang::Core | Lang::En | Lang::Eu) && s1.canonical == "en+eu", "en+eu failed");
    }
    {
        const FilterSpec s_ja = vocab_filter::parse_filter("en+jp");
        STRIX_CHECK(s_ja.langs == (Lang::Core | Lang::En | Lang::Ja | Lang::Zh) && s_ja.canonical == "en+zh+ja",
                    "en+jp failed: ", s_ja.canonical);
        const FilterSpec s_zh = vocab_filter::parse_filter("en+cn+kr+fa+other");
        STRIX_CHECK(s_zh.langs == (Lang::Core | Lang::En | Lang::Zh | Lang::Ko | Lang::Ar | Lang::Other) &&
                        s_zh.canonical == "en+zh+ko+ar+other",
                    "multi aliases failed: ", s_zh.canonical);
    }

    // Must fail when missing mandatory 'en'/'core'
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("ru"); }, "core"), "ru alone must fail");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("eu"); }, "core"), "eu alone must fail");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("core"); }, "must include 'en'"), "core alone must fail");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("cyrillic"); }, "must include 'en'"),
                "cyrillic alone must fail");

    // Must reject comma delimiter, removed options, invalid mixes, unknown tags
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("en,ru"); }, "'+' instead of ','"),
                "comma must be rejected");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("no-cjk"); }, "unknown filter tag"),
                "no-cjk must be rejected");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("file:/tmp/foo"); }, "unknown filter tag"),
                "file: must be rejected");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("all+ru"); }, "cannot be combined"),
                "all+ru must be rejected");
    STRIX_CHECK(throws_containing([] { vocab_filter::parse_filter("en+"); }, "empty tag"), "trailing + must fail");
}

void test_mtp_vocab_removed() {
    const std::string cfg_path = "/tmp/test_vocab_filter_removed.conf";
    {
        std::ofstream f(cfg_path);
        f << "mtp-vocab = 0\n";
    }
    STRIX_CHECK(throws_containing(
                    [&] {
                        ServerSettings s(fastasfuck_server_options("/models"));
                        s.load({"--config", cfg_path});
                    },
                    "mtp-vocab-filter"),
                "mtp-vocab in config must fail with migration hint");
    std::remove(cfg_path.c_str());

    STRIX_CHECK(throws_containing(
                    [] {
                        ServerSettings s(fastasfuck_server_options("/models"));
                        s.load({"--mtp-vocab", "65536"});
                    },
                    "mtp-vocab-filter"),
                "--mtp-vocab on CLI must fail with migration hint");
}

void test_synthetic_classification_and_wall() {
    // Build a minimal valid BPE tokenizer JSON with ~12k tokens:
    //   [0..255]: single bytes
    //   [256..1023]: Epoch-1 English + Core + audit edge cases (BOM, partial UTF-8, Greek math/words, Cyrillic, etc.)
    //   [1024..10239]: Han wall (9,216 Han tokens across 9 blocks of 1024 -> triggers >=8-block auto wall!)
    //   [10240..10399]: Epoch-3 eu-ascii, European diacritics, Russian, Arabic, Kana, Hangul, Vietnamese, Thai
    //   added: <think>, </think>, <tool_call>, <|im_end|>
    std::vector<std::string> vocab_raw;
    vocab_raw.reserve(10400);
    for (int b = 0; b < 256; ++b) vocab_raw.emplace_back(1, static_cast<char>(b));

    const int id_en_word = (int)vocab_raw.size();
    vocab_raw.push_back("hello");
    const int id_bom_using = (int)vocab_raw.size();
    vocab_raw.push_back("\xEF\xBB\xBF" "using");  // U+FEFF + using (must NOT be Arabic!)
    const int id_part_ru = (int)vocab_raw.size();
    vocab_raw.push_back(" \xD0");  // space + Cyrillic lead byte (must map to Ru, not Core!)
    const int id_part_punct = (int)vocab_raw.size();
    vocab_raw.push_back(" \xE2\x80");  // space + smart quote/dash prefix (must stay Core!)
    const int id_part_ar = (int)vocab_raw.size();
    vocab_raw.push_back(" \xD8");  // space + Arabic lead byte
    const int id_greek_math = (int)vocab_raw.size();
    vocab_raw.push_back(" \xCF\x80");  // ' π' -> single Greek math letter -> Core!
    const int id_greek_word = (int)vocab_raw.size();
    vocab_raw.push_back("\xCE\xBB\xCE\xBF\xCE\xB3");  // 'λογ' -> Other, NOT Core!
    const int id_indent = (int)vocab_raw.size();
    vocab_raw.push_back("    ");

    while ((int)vocab_raw.size() < 1024) {
        vocab_raw.push_back("tok" + std::to_string(vocab_raw.size()));
    }
    // 9,216 distinct Han tokens [1024..10239] using valid CJK Unified Ideographs [0x4E00..0x9FFF]
    while ((int)vocab_raw.size() < 10240) {
        const uint32_t cp = 0x4E00u + static_cast<uint32_t>(vocab_raw.size() - 1024);
        vocab_raw.push_back(cp_to_utf8(cp));
    }
    // Epoch 3 tokens [10240..]
    const int id_eu_ascii = (int)vocab_raw.size();
    vocab_raw.push_back("deshalb");
    const int id_eu_ext = (int)vocab_raw.size();
    vocab_raw.push_back("caf\xC3\xA9");  // café
    const int id_ru_word = (int)vocab_raw.size();
    vocab_raw.push_back("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");  // привет
    const int id_ar_word = (int)vocab_raw.size();
    vocab_raw.push_back("\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7");  // مرحبا
    const int id_kana_word = (int)vocab_raw.size();
    vocab_raw.push_back("\xE3\x81\x93\xE3\x82\x93");  // こん
    const int id_ko_word = (int)vocab_raw.size();
    vocab_raw.push_back("\xED\x95\x9C\xEA\xB8\x80");  // 한글
    const int id_vi_word = (int)vocab_raw.size();
    vocab_raw.push_back("ti\xE1\xBA\xBFng");  // tiếng (Vietnamese -> Other, NOT Eu)
    while ((int)vocab_raw.size() < 11300) {
        vocab_raw.push_back("eow" + std::to_string(vocab_raw.size()));
    }

    const int n_vocab = (int)vocab_raw.size();
    const std::vector<std::string> added = {"<|im_end|>", "<think>", "</think>", "<tool_call>"};

    const std::string tok_path = "/tmp/test_vocab_filter_tok.json";
    {
        std::ofstream f(tok_path, std::ios::binary);
        f << "{\n"
          << "  \"normalizer\": {\"type\": \"NFC\"},\n"
          << "  \"pre_tokenizer\": {\"type\": \"Sequence\", \"pretokenizers\": [\n"
          << "    {\"type\": \"Split\", \"pattern\": {\"Regex\": "
             "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\""
             "}, \"behavior\": \"Isolated\", \"invert\": false},\n"
          << "    {\"type\": \"ByteLevel\"}\n"
          << "  ]},\n"
          << "  \"decoder\": {\"type\": \"ByteLevel\"},\n"
          << "  \"added_tokens\": [\n";
        for (size_t i = 0; i < added.size(); ++i) {
            f << "    {\"id\": " << (n_vocab + (int)i) << ", \"content\": \"" << json_escape(added[i])
              << "\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false}"
              << (i + 1 < added.size() ? "," : "") << "\n";
        }
        f << "  ],\n"
          << "  \"model\": {\"type\": \"BPE\", \"unk_token\": null, \"byte_fallback\": false, "
             "\"ignore_merges\": false, \"continuing_subword_prefix\": \"\", \"end_of_word_suffix\": \"\", "
             "\"dropout\": null, \"merges\": [], \"vocab\": {\n";
        for (int i = 0; i < n_vocab; ++i) {
            f << "    \"" << json_escape(bytes_to_gpt2_text(vocab_raw[(size_t)i])) << "\": " << i
              << (i + 1 < n_vocab ? "," : "") << "\n";
        }
        f << "  }}\n}\n";
    }

    Tokenizer tok(tok_path);
    std::remove(tok_path.c_str());

    auto as_set = [](const std::vector<int32_t> &v) { return std::unordered_set<int32_t>(v.begin(), v.end()); };

    // Test 'en'
    int64_t eu_start = 0;
    const auto en_ids = vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en"), tok.size(), &eu_start);
    STRIX_CHECK(eu_start == id_eu_ascii, "expected eu_ascii_start=", id_eu_ascii, ", got ", eu_start);
    const auto en_s = as_set(en_ids);
    STRIX_CHECK(std::is_sorted(en_ids.begin(), en_ids.end()), "filtered IDs must stay sorted");
    STRIX_CHECK(en_s.count(id_en_word) && en_s.count(id_indent), "en word & indent kept");
    STRIX_CHECK(en_s.count(id_bom_using), "U+FEFF BOM + using MUST stay in en (not Arabic!)");
    STRIX_CHECK(en_s.count(id_part_punct), "' \\xe2\\x80' punct fragment MUST stay in Core/en");
    STRIX_CHECK(en_s.count(id_greek_math), "single Greek math letter 'π' MUST stay in Core/en");
    STRIX_CHECK(!en_s.count(id_greek_word), "multi-letter Greek word must NOT enter en");
    STRIX_CHECK(!en_s.count(id_part_ru) && !en_s.count(id_part_ar), "Ru/Ar partial fragments excluded from en");
    STRIX_CHECK(!en_s.count(id_eu_ascii) && !en_s.count(id_eu_ext), "post-wall eu_ascii & diacritics excluded from en");
    for (int i = 0; i < (int)added.size(); ++i) {
        STRIX_CHECK(en_s.count(n_vocab + i), "added special token id ", n_vocab + i, " MUST stay in en");
    }

    // Test 'en+ru'
    const auto ru_ids = vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+ru"), tok.size());
    const auto ru_s = as_set(ru_ids);
    STRIX_CHECK(ru_s.count(id_ru_word) && ru_s.count(id_part_ru), "en+ru keeps Cyrillic word & ' \\xd0' fragment");
    STRIX_CHECK(!ru_s.count(id_eu_ascii) && !ru_s.count(id_eu_ext), "en+ru keeps eu excluded");
    STRIX_CHECK(!ru_s.count(id_kana_word) && !ru_s.count(1024), "en+ru excludes Han/Kana");

    // Test 'en+eu'
    const auto eu_ids = vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+eu"), tok.size());
    const auto eu_s = as_set(eu_ids);
    STRIX_CHECK(eu_s.count(id_eu_ascii) && eu_s.count(id_eu_ext), "en+eu keeps eu_ascii & café");
    STRIX_CHECK(!eu_s.count(id_vi_word), "en+eu excludes Vietnamese 'tiếng' (belongs to other)");

    // Test 'en+ja' (implies Zh Han too!)
    const auto ja_ids = vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+ja"), tok.size());
    const auto ja_s = as_set(ja_ids);
    STRIX_CHECK(ja_s.count(id_kana_word) && ja_s.count(1024), "en+ja keeps Kana and Kanji (Han)");
    STRIX_CHECK(!ja_s.count(id_ko_word), "en+ja excludes Hangul");

    // Test 'en+ko', 'en+ar', 'en+other'
    STRIX_CHECK(as_set(vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+ko"), tok.size())).count(id_ko_word),
                "en+ko keeps Hangul");
    const auto ar_s = as_set(vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+ar"), tok.size()));
    STRIX_CHECK(ar_s.count(id_ar_word) && ar_s.count(id_part_ar), "en+ar keeps Arabic word & ' \\xd8'");
    const auto oth_s = as_set(vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en+other"), tok.size()));
    STRIX_CHECK(oth_s.count(id_vi_word) && oth_s.count(id_greek_word), "en+other keeps Vietnamese & Greek words");

    // Cache key checks
    const std::string ck_all = vocab_filter::cache_key(vocab_filter::parse_filter("all"), {});
    const std::string ck_en = vocab_filter::cache_key(vocab_filter::parse_filter("en"), en_ids);
    const std::string ck_ru = vocab_filter::cache_key(vocab_filter::parse_filter("en+ru"), ru_ids);
    STRIX_CHECK(ck_all == "all", "all cache_key mismatch");
    STRIX_CHECK(ck_en != ck_ru && ck_en.rfind("en#", 0) == 0 && ck_ru.rfind("en+ru#", 0) == 0,
                "filtered cache_key mismatch: ", ck_en, " vs ", ck_ru);
}

void dump_live_tokenizer(const std::string &path) {
    Tokenizer tok(path);
    std::cout << "Live tokenizer: " << path << " (" << tok.size() << " tokens)\n";
    int64_t eu_start = 0;
    const auto en_ids = vocab_filter::filter_vocabulary(tok, vocab_filter::parse_filter("en"), tok.size(), &eu_start);
    std::cout << "Detected eu_ascii_start = " << eu_start << "\n\n";
    for (const char *spec_s : {"en", "en+ru", "en+eu", "en+zh", "en+ja", "en+ko", "en+ar", "en+other", "en+eu+ru", "all"}) {
        const FilterSpec spec = vocab_filter::parse_filter(spec_s);
        if (vocab_filter::is_all(spec)) {
            std::cout << std::left << std::setw(12) << spec_s << " -> 248320 rows (~357.6 MB Q4) | key=all\n";
            continue;
        }
        const auto ids = vocab_filter::filter_vocabulary(tok, spec, tok.size());
        const double mb = static_cast<double>(ids.size()) * 1440.0 / 1e6;
        std::cout << std::left << std::setw(12) << spec_s
                  << " | canonical=" << std::setw(12) << spec.canonical
                  << " | rows=" << std::setw(7) << ids.size()
                  << " (~" << std::fixed << std::setprecision(1) << std::setw(5) << mb << " MB Q4)"
                  << " | key=" << vocab_filter::cache_key(spec, ids) << "\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    test_parse_filter();
    test_mtp_vocab_removed();
    test_synthetic_classification_and_wall();
    std::printf("PASS: test_vocab_filter\n");
    if (argc >= 2) {
        dump_live_tokenizer(argv[1]);
    }
    return 0;
}
