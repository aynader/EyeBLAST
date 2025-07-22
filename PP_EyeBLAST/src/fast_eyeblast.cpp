#include "fast_eyeblast.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <limits>
#include <climits>
#include <cctype>
#include <stdexcept> 

namespace pp_eyeblast {


const uint64_t MASK64 = UINT64_MAX;


inline int popcount64(uint64_t value) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value) {
        count += value & 1;
        value >>= 1;
    }
    return count;
#endif
}


double DeterministicRandom::random() {
    
    state_ = (state_ * 1103515245ULL + 12345ULL) & 0x7fffffffULL;
    return static_cast<double>(state_) / 0x80000000ULL;
}

bool DeterministicRandom::random_bool() {
    
    return random() < 0.5;
}

uint64_t DeterministicRandom::random_uint64() {
    uint64_t result = 0;
    for (int i = 0; i < 64; ++i) {
        
        if (random_bool()) {
            result |= (1ULL << i);
        }
    }
    return result;
}


FastEyeBLAST::FastEyeBLAST(int seed_length, int band_width, 
                           bool use_duration_bins, int duration_bins,
                           int lsh_hash_size)
    : seed_length_(seed_length), band_width_(band_width),
      use_duration_bins_(use_duration_bins), duration_bins_(duration_bins),
      next_token_id_(0), frozen_(false), bits_per_token_(0), tokens_per_word_(0), base_(31),
      lsh_hash_size_(lsh_hash_size) {
}

static inline int deterministic_token_id(const std::string& tok) {
    if (tok.empty())
        return 0;

    char c = tok[0];
    int letter = (c == 'M') ? 0
               : (c == 'F') ? 1
               :              2;          

    int value = 0;
    for (size_t i = 1; i < tok.size(); ++i)
        if (std::isdigit(tok[i]))
            value = value * 10 + (tok[i] - '0');   

    value &= 0x1FF;                     
    return (letter << 9) | value;       
}


int FastEyeBLAST::token_to_integer(const std::string& token) {
    if (token_to_id_.find(token) == token_to_id_.end()) {
        if (frozen_) {
            throw std::runtime_error("Cannot add new token '" + token + "' - dictionary is frozen");
        }
        token_to_id_[token] = next_token_id_;
        id_to_token_[next_token_id_] = token;
        next_token_id_++;
    }
    return token_to_id_[token];
}

std::string FastEyeBLAST::bin_duration(const std::string& token) {
    if (!use_duration_bins_ || token.empty()) {
        return token;
    }
    
    char token_type = token[0];
    std::string duration_str = token.substr(1);
    
    if (duration_str.empty() || 
        !std::all_of(duration_str.begin(), duration_str.end(), ::isdigit)) {
        return token;
    }
    
    try {
        int duration = std::stoi(duration_str);
        int bin_idx;
        
        if (duration == 0) {
            bin_idx = 0;
        } else {
            bin_idx = std::min(static_cast<int>(std::log2(duration + 1)), 
                              duration_bins_ - 1);
        }
        
        return std::string(1, token_type) + std::to_string(bin_idx);
    } catch (...) {
        return token;
    }
}


void FastEyeBLAST::initialize_bit_packing() {
    int alphabet_size = token_to_id_.size();
    
    bits_per_token_ = std::max(1, static_cast<int>(std::ceil(std::log2(alphabet_size))));
    tokens_per_word_ = 64 / bits_per_token_;
    
    std::cout << "Alphabet size: " << alphabet_size << std::endl;
    std::cout << "Bits per token: " << bits_per_token_ << std::endl;
    std::cout << "Tokens per 64-bit word: " << tokens_per_word_ << std::endl;
    
    
    base_ = 31;  
    pow_base_.clear();
    pow_base_.push_back(1);
    for (int i = 1; i < std::max(seed_length_, 100); i++) {
        pow_base_.push_back((pow_base_.back() * base_) & MASK64);
    }
}


void FastEyeBLAST::initialize_simhash() {
    int alphabet_size = token_to_id_.size();
    id_bitmasks_.clear();
    id_bitmasks_.resize(alphabet_size);
    
    
    DeterministicRandom rng(42);
    
    for (int i = 0; i < alphabet_size; i++) {
        uint64_t mask = 0;
        for (int j = 0; j < lsh_hash_size_; j++) {
            
            if (rng.random() < 0.5) {
                mask |= (1ULL << j);
            }
        }
        id_bitmasks_[i] = mask;
    }
}

std::vector<uint64_t> FastEyeBLAST::pack_ids_into_words(const std::vector<int>& id_list) {
    if (id_list.empty()) return {};
    
    int num_words = (id_list.size() + tokens_per_word_ - 1) / tokens_per_word_;
    std::vector<uint64_t> packed_words(num_words, 0);
    
    for (int i = 0; i < num_words; i++) {
        uint64_t word = 0;
        for (int j = 0; j < tokens_per_word_; j++) {
            int idx = i * tokens_per_word_ + j;
            if (idx < id_list.size()) {
                uint64_t token_bits = id_list[idx] & ((1ULL << bits_per_token_) - 1);
                word |= token_bits << (j * bits_per_token_);
            }
        }
        packed_words[i] = word;
    }
    
    return packed_words;
}


std::vector<uint64_t> FastEyeBLAST::calculate_rolling_hash(const std::vector<int>& id_list) {
    if (id_list.size() < seed_length_) return {};
    
    std::vector<uint64_t> hashes;
    int k = seed_length_;
    
    
    uint64_t h = 0;
    for (int i = 0; i < k; i++) {
        h = (h * base_ + id_list[i]) & MASK64;
    }
    hashes.push_back(h);
    
    
    for (size_t i = k; i < id_list.size(); i++) {
        
        h = (h - id_list[i - k] * pow_base_[k - 1]) & MASK64;
        
        h = (h * base_) & MASK64;
        
        h = (h + id_list[i]) & MASK64;
        hashes.push_back(h);
    }
    
    return hashes;
}


uint64_t FastEyeBLAST::compute_simhash(const std::vector<int>& id_list) {
    if (id_bitmasks_.empty()) {
        initialize_simhash();
    }
    
    
    std::vector<int> counters(lsh_hash_size_, 0);
    
    
    for (int token_id : id_list) {
        if (token_id >= 0 && token_id < static_cast<int>(id_bitmasks_.size())) {
            uint64_t bitmask = id_bitmasks_[token_id];
            for (int i = 0; i < lsh_hash_size_; i++) {
                if (bitmask & (1ULL << i)) {
                    counters[i]++;
                } else {
                    counters[i]--;
                }
            }
        }
    }
    
    
    uint64_t fingerprint = 0;
    for (int i = 0; i < lsh_hash_size_; i++) {
        if (counters[i] > 0) {
            fingerprint |= (1ULL << i);
        }
    }
    
    return fingerprint;
}
int FastEyeBLAST::hamming_distance(uint64_t a, uint64_t b) {
    return popcount64(a ^ b);
}

std::vector<SeedPair> FastEyeBLAST::extract_seeds(const std::vector<uint64_t>& packed1, 
                                                  const std::vector<uint64_t>& packed2) {
    std::vector<SeedPair> seeds;
    int max_mismatches = 1;
    
    int min_len = std::min(packed1.size(), packed2.size());
    for (int i = 0; i < min_len; i++) {
        uint64_t xor_result = packed1[i] ^ packed2[i];
        int mismatches = popcount64(xor_result);
        
        if (mismatches <= max_mismatches * bits_per_token_) {
            int pos1 = i * tokens_per_word_;
            int pos2 = i * tokens_per_word_;
            seeds.emplace_back(pos1, pos2);
        }
    }
    
    return seeds;
}

std::vector<SeedPair> FastEyeBLAST::greedy_chain_seeds(const std::vector<SeedPair>& seeds,
                                                       int len1, int len2) {
    if (seeds.empty()) return {};
    
    
    std::map<int, std::vector<SeedPair>> diagonals;
    for (const auto& seed : seeds) {
        int diag = seed.pos2 - seed.pos1;
        diagonals[diag].push_back(seed);
    }
    
    
    auto best_diag = std::max_element(diagonals.begin(), diagonals.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second.size() < b.second.size();
                                     });
    
    std::vector<SeedPair> best_seeds = best_diag->second;
    std::sort(best_seeds.begin(), best_seeds.end(),
              [](const SeedPair& a, const SeedPair& b) {
                  return a.pos1 < b.pos1;
              });
    
    
    std::vector<SeedPair> chain;
    if (!best_seeds.empty()) {
        chain.push_back(best_seeds[0]);
        for (size_t i = 1; i < best_seeds.size(); i++) {
            if (best_seeds[i].pos1 > chain.back().pos1 && 
                best_seeds[i].pos2 > chain.back().pos2) {
                chain.push_back(best_seeds[i]);
            }
        }
    }
    
    return chain;
}

MatchResult FastEyeBLAST::banded_levenshtein(const std::vector<int>& seq1,
                                           const std::vector<int>& seq2,
                                           int band_width,
                                           const std::vector<SeedPair>& chain) {
    if (chain.empty()) {
        
        return MatchResult(static_cast<int>(-(seq1.size() + seq2.size())), {});
    }
    
    
    int diag = chain[0].pos2 - chain[0].pos1;
    
    
    int half_band = band_width / 2;
    
    
    int m = seq1.size();
    int n = seq2.size();
    
    
    int match_score = 2;
    int mismatch_penalty = -1;
    int gap_penalty = -1;
    
    
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(band_width, 0));
    
    
    for (int i = 0; i < band_width; i++) {
        dp[0][i] = i * gap_penalty;
    }
    
    for (int i = 1; i <= m; i++) {
        
        int col_start = std::max(1, i + diag - half_band);
        int col_end = std::min(n + 1, i + diag + half_band + 1);
        int band_start = 0;
        
        
        if (col_start > 1) {
            dp[i][0] = dp[i-1][0] + gap_penalty;
        } else {
            dp[i][0] = i * gap_penalty;
        }
        
        
        for (int j = col_start; j < col_end; j++) {
            int band_j = j - col_start + band_start;
            
            if (band_j >= band_width) {
                break;
            }
            
            
            if (j <= n) {
                bool match = (seq1[i-1] == seq2[j-1]);
                
                int col_start_prev = std::max(1, (i-1) + diag - half_band);
                int band_j_prev    = (j-1) - col_start_prev;
                int diag_score     = (band_j_prev >= 0 && band_j_prev < band_width)? dp[i-1][band_j_prev] + (match ? match_score: mismatch_penalty): INT_MIN/2;
                int up_score = dp[i-1][band_j] + gap_penalty;
                int left_score = (band_j > 0) ? dp[i][band_j-1] : INT_MIN;
                
                
                dp[i][band_j] = std::max({diag_score, up_score, left_score});
            }
        }
    }
    
    
    
    int final_col_start = std::max(1, m + diag - half_band);
    int final_band_pos = n - final_col_start;
    
    
    if (final_band_pos < 0) final_band_pos = 0;
    if (final_band_pos >= band_width) final_band_pos = band_width - 1;
    
    int score = dp[m][final_band_pos];

    
    
    return MatchResult(score, chain);
}

Encoding FastEyeBLAST::encode_and_pack(const std::vector<std::string>& tokens) {
    
    std::vector<int> encoded;
    encoded.reserve(tokens.size());
    for (const std::string& tok : tokens) {
        encoded.push_back(token_to_integer(bin_duration(tok)));
    }

    
    if (bits_per_token_ == 0)
        initialize_bit_packing();

    
    std::vector<uint64_t> packed  = pack_ids_into_words(encoded);
    uint64_t              simhash = compute_simhash(encoded);

    
    Encoding enc;
    enc.ids             = std::move(encoded);
    enc.packed          = std::move(packed);
    enc.simhash         = simhash;
    enc.bits_per_token  = bits_per_token_;   
    enc.tokens_per_word = tokens_per_word_;  
    return enc;
}

double FastEyeBLAST::normalized_similarity(const std::vector<int>& seq1,
                                         const std::vector<int>& seq2,
                                         int band_width) {
    
    std::vector<uint64_t> packed1 = pack_ids_into_words(seq1);
    std::vector<uint64_t> packed2 = pack_ids_into_words(seq2);
    
    
    std::vector<SeedPair> seeds = extract_seeds(packed1, packed2);
    std::cout << "DEBUG: Found " << seeds.size() << " seeds" << std::endl;
    
    
    std::vector<SeedPair> chain = greedy_chain_seeds(seeds, seq1.size(), seq2.size());
    std::cout << "DEBUG: Chain has " << chain.size() << " seeds" << std::endl;
    
    
    MatchResult result = banded_levenshtein(seq1, seq2, band_width, chain);
    std::cout << "DEBUG: Raw score = " << result.score << std::endl;
    
    
    int max_possible_score = std::max(seq1.size(), seq2.size()) * 2;
    std::cout << "DEBUG: Max possible score = " << max_possible_score << std::endl;
    double normalized_score = std::max(0.0, static_cast<double>(result.score)) / max_possible_score;
    std::cout << "DEBUG: Normalized score = " << normalized_score << std::endl;
    return normalized_score;
}


static FastEyeBLAST global_instance;

FastEyeBLAST &get_global_instance() {
    return global_instance;
}

Encoding encode_and_pack(const std::vector<std::string>& tokens) {
    return global_instance.encode_and_pack(tokens);
}

double normalized_similarity(const std::vector<int>& seq1,
                            const std::vector<int>& seq2,
                            int band_width) {
    return global_instance.normalized_similarity(seq1, seq2, band_width);
}


void FastEyeBLAST::freeze_dictionary() {
    frozen_ = true;
    std::cout << "Dictionary frozen with " << token_to_id_.size() << " tokens" << std::endl;
}

bool FastEyeBLAST::is_frozen() const {
    return frozen_;
}

void FastEyeBLAST::reset_dictionary() {
    token_to_id_.clear();
    id_to_token_.clear();
    next_token_id_ = 0;
    frozen_ = false;
    bits_per_token_ = 0;
    tokens_per_word_ = 0;
    id_bitmasks_.clear();
}

} 