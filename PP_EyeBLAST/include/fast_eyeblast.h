#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <unordered_map>

namespace pp_eyeblast {

struct Encoding {
    std::vector<int> ids;
    std::vector<uint64_t> packed;
    uint64_t simhash;
    int bits_per_token;
    int tokens_per_word;
    
    Encoding() : simhash(0) {}
    Encoding(const std::vector<int>& ids, const std::vector<uint64_t>& packed, uint64_t simhash)
        : ids(ids), packed(packed), simhash(simhash) {}
};

struct SeedPair {
    int pos1;
    int pos2;
    
    SeedPair(int p1, int p2) : pos1(p1), pos2(p2) {}
};

struct MatchResult {
    int score;
    std::vector<SeedPair> alignment;
    
    MatchResult(int s, const std::vector<SeedPair>& a) : score(s), alignment(a) {}
};

class FastEyeBLAST {
private:
    
    int seed_length_;
    int band_width_;
    bool use_duration_bins_;
    int duration_bins_;
    
    
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    int next_token_id_;
    bool frozen_;
    
    
    int bits_per_token_;
    int tokens_per_word_;
    uint64_t base_;
    std::vector<uint64_t> pow_base_;
    
    
    std::vector<uint64_t> id_bitmasks_;
    int lsh_hash_size_;
    
    
    void initialize_bit_packing();
    void initialize_simhash();
    std::vector<uint64_t> pack_ids_into_words(const std::vector<int>& id_list);
    std::vector<uint64_t> calculate_rolling_hash(const std::vector<int>& id_list);
    std::vector<SeedPair> extract_seeds(const std::vector<uint64_t>& packed1, 
                                       const std::vector<uint64_t>& packed2);
    std::vector<SeedPair> greedy_chain_seeds(const std::vector<SeedPair>& seeds, 
                                            int len1, int len2);
    MatchResult banded_levenshtein(const std::vector<int>& seq1, 
                                  const std::vector<int>& seq2,
                                  int band_width, 
                                  const std::vector<SeedPair>& chain);

public:
    FastEyeBLAST(int seed_length = 3, int band_width = 5, 
                 bool use_duration_bins = true, int duration_bins = 8,
                 int lsh_hash_size = 64);
    
    ~FastEyeBLAST() = default;
    
    
    Encoding encode_and_pack(const std::vector<std::string>& tokens);
    
    double normalized_similarity(const std::vector<int>& seq1, 
                                const std::vector<int>& seq2, 
                                int band_width);

    
    uint64_t compute_simhash(const std::vector<int>& id_list);
    int hamming_distance(uint64_t a, uint64_t b);
    
    
    const std::unordered_map<std::string, int>& get_token_to_id() const { return token_to_id_; }
    int get_bits_per_token() const { return bits_per_token_; }
    int get_tokens_per_word() const { return tokens_per_word_; }
    
    
    std::string bin_duration(const std::string& token);
    int token_to_integer(const std::string& token);
    
    
    void freeze_dictionary();
    bool is_frozen() const;
    void reset_dictionary();  
};


class DeterministicRandom {
private:
    uint64_t state_;
    
public:
    explicit DeterministicRandom(uint64_t seed = 42) : state_(seed) {}
    
    void seed(uint64_t s) { state_ = s; }
    
    double random();
    bool random_bool();
    uint64_t random_uint64();
};


Encoding encode_and_pack(const std::vector<std::string>& tokens);
double normalized_similarity(const std::vector<int>& seq1, 
                            const std::vector<int>& seq2, 
                            int band_width = 5);


FastEyeBLAST &get_global_instance();

} 