#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <emp-tool/emp-tool.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include <nlohmann/json.hpp>
#include "fast_eyeblast.h"
#include "gc_utils.h"

using namespace emp;
using namespace pp_eyeblast;



std::vector<std::pair<SeedPair, Bit>> SeedExtractionCircuit(const std::vector<Integer>& words_alice,
                                                             const std::vector<Integer>& words_bob,
                                                             int bits_per_token,
                                                             int tokens_per_word) {
    
    std::vector<std::pair<SeedPair, Bit>> seeds_with_validity;
    int max_mismatches = 1;  
    size_t min_len = std::min(words_alice.size(), words_bob.size());
    
    std::cout << "EXACT original seed extraction from " << min_len << " word pairs..." << std::endl;
    
    for (size_t i = 0; i < min_len; i++) {
        
        Integer xor_result = words_alice[i] ^ words_bob[i];
        
        
        std::vector<Bit> diff_bits(64);
        for (int j = 0; j < 64; j++) {
            diff_bits[j] = xor_result[j];
        }
        
        Integer mismatch_count(64, 0, PUBLIC);
        for (int j = 0; j < 64; j++) {
            Integer bit_val(64, 0, PUBLIC);
            bit_val[0] = diff_bits[j];
            mismatch_count = mismatch_count + bit_val;
        }
        
        
        Integer threshold(64, max_mismatches * bits_per_token, PUBLIC);
        Bit is_seed = threshold.geq(mismatch_count);
        
        
        int pos1 = i * tokens_per_word;
        int pos2 = i * tokens_per_word;
        seeds_with_validity.emplace_back(SeedPair(pos1, pos2), is_seed);
    }
    
    std::cout << "Found " << seeds_with_validity.size() << " potential seeds" << std::endl;
    return seeds_with_validity;
}



std::vector<SeedPair> GreedyChainSeedsCircuit(const std::vector<std::pair<SeedPair, Bit>>& seeds_with_validity,
                                              int len1, int len2) {
    if (seeds_with_validity.empty()) return {};
    
    
    std::vector<SeedPair> valid_seeds;
    std::cout << "=== DEBUG: FILTERING SEEDS ===" << std::endl;
    for (const auto& seed_pair : seeds_with_validity) {
        bool is_valid = seed_pair.second.reveal<bool>();
        std::cout << "Seed pos1=" << seed_pair.first.pos1 << ", pos2=" << seed_pair.first.pos2 << ", valid=" << is_valid << std::endl;
        if (is_valid) {
            valid_seeds.push_back(seed_pair.first);
        }
    }
    std::cout << "Valid seeds found: " << valid_seeds.size() << " out of " << seeds_with_validity.size() << std::endl;
    
    if (valid_seeds.empty()) return {};
    
    
    std::map<int, std::vector<SeedPair>> diagonals;
    for (const auto& seed : valid_seeds) {
        int diag = seed.pos2 - seed.pos1;
        diagonals[diag].push_back(seed);
    }
    
    
    auto best_diag = std::max_element(diagonals.begin(), diagonals.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second.size() < b.second.size();
                                     });
    
    std::cout << "=== DEBUG: DIAGONAL SELECTION ===" << std::endl;
    for (const auto& [diag, seeds] : diagonals) {
        std::cout << "Diagonal " << diag << ": " << seeds.size() << " seeds" << std::endl;
    }
    std::cout << "Selected diagonal: " << best_diag->first << " with " << best_diag->second.size() << " seeds" << std::endl;
    
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
    
    std::cout << "Chained " << chain.size() << " seeds" << std::endl;
    return chain;
}


Integer BandedDPCircuit(const std::vector<Integer>& seq1_tokens,
                        const std::vector<Integer>& seq2_tokens,
                        size_t                     len1,          
                        size_t                     len2,          
                        int                        band_width,
                        const std::vector<SeedPair>& chain,
                        int                        party) {

    std::cout << "Computing EXACT original token-level banded Levenshtein DP with band width "
              << band_width << "...\n";

    const size_t m = len1;   
    const size_t n = len2;
    std::cout << "Token lengths (real): Alice=" << m << ", Bob=" << n << std::endl;
    
    
    std::cout << "=== DEBUG: FIRST 10 TOKENS ===" << std::endl;
    std::cout << "Alice tokens: ";
    for (int i = 0; i < std::min(10, static_cast<int>(m)); i++) {
        std::cout << seq1_tokens[i].reveal<int>() << " ";
    }
    std::cout << std::endl;
    std::cout << "Bob tokens: ";
    for (int i = 0; i < std::min(10, static_cast<int>(n)); i++) {
        std::cout << seq2_tokens[i].reveal<int>() << " ";
    }
    std::cout << std::endl;
    
    
    std::cout << "=== DEBUG: TOKEN DICTIONARY STATE ===" << std::endl;
    std::cout << "First few token mappings:" << std::endl;
    auto raw_tokens = std::vector<std::string>{"F1", "F2", "F3", "M1", "M2", "M3", "S1", "S2", "S3"};
    for (const auto& token : raw_tokens) {
        std::cout << "  " << token << " -> " << get_global_instance().token_to_integer(token) << std::endl;
    }

    
    if (chain.empty()) {
        return Integer(32, static_cast<int>((m + n)), PUBLIC);   
    }

    
    
    int diag = chain[0].pos2 - chain[0].pos1;
    int half_band = band_width / 2;

    std::cout << "=== BANDED DP DEBUG ===\n"
              << "Band width: " << band_width << " (input parameter)\n"
              << "Half band: "  << half_band   << '\n'
              << "Diagonal from chain[0]: " << diag << " (from " << chain.size() << " seeds)\n";

    
    const Integer match_score(32,  2,  PUBLIC);
    const Integer mismatch_pen(32, -1, PUBLIC);
    const Integer gap_penalty(32,  -1, PUBLIC);

    
    std::vector<std::vector<Integer>> dp(m + 1,
                                         std::vector<Integer>(band_width,
                                                              Integer(32, 0, PUBLIC)));

    for (int j = 0; j < band_width; ++j)
        dp[0][j] = Integer(32, j, PUBLIC) * gap_penalty;

    
    for (size_t i = 1; i <= m; ++i) {
        
        
        bool should_debug = (i <= 5) || (i >= m-5) || (i % 200 == 0);
        if (should_debug) {
            std::cout << "=== DEBUG: DP ROW " << i << " (Party " << (party == ALICE ? "Alice" : "Bob") << ") ===" << std::endl;
        }

        int col_start = std::max(1, static_cast<int>(i) + diag - half_band);
        int col_end   = std::min(static_cast<int>(n) + 1,
                                 static_cast<int>(i) + diag + half_band + 1);

        
        dp[i][0] = (col_start > 1) ? (dp[i-1][0] + gap_penalty)
                                   : Integer(32, static_cast<int>(i), PUBLIC) * gap_penalty;

        for (int j = col_start; j < col_end; ++j) {
            int band_j = j - col_start;          
            if (band_j >= band_width) break;
            
            
            if (i <= 5 && j <= 10) {
                std::cout << "Row " << i << ": col_start=" << col_start << ", col_end=" << col_end 
                          << ", j=" << j << ", band_j=" << band_j << std::endl;
            }

            
            int band_j_prev = (band_j > 0) ? band_j - 1 : 0;
            Integer diag_base = dp[i-1][band_j_prev];

            
            Bit      is_match   = seq1_tokens[i-1].equal(seq2_tokens[j-1]);
            Integer  mismatch_pen(32, -1, PUBLIC);  
            Integer  match_score(32, 2, PUBLIC);   
            Integer  match_delta = mismatch_pen.select(is_match, match_score);

            Integer diag_score  = diag_base + match_delta;
            Integer up_score    = dp[i-1][band_j] + gap_penalty;
            Integer left_score  = (band_j ? dp[i][band_j-1] + gap_penalty
                                          : Integer(32, INT_MIN/2, PUBLIC));

            
            Integer best = diag_score;
            Bit     better = up_score.geq(best);
            Integer tmp(32,0,PUBLIC); tmp[0] = better;
            best = best + tmp * (up_score - best);

            better = left_score.geq(best);
            tmp[0] = better;
            best = best + tmp * (left_score - best);

            dp[i][band_j] = best;
            
            
            bool detailed_debug = (i <= 20 && j <= 20) || (i >= m-5 && j >= n-5) || (i % 200 == 0 && j % 10 == 0);
            if (detailed_debug) {
                bool match_result = is_match.reveal<bool>();
                int token1 = seq1_tokens[i-1].reveal<int>();
                int token2 = seq2_tokens[j-1].reveal<int>();
                int delta = match_delta.reveal<int>();
                int diag_val = diag_base.reveal<int>();
                int up_val = up_score.reveal<int>();
                int left_val = left_score.reveal<int>();
                int best_val = best.reveal<int>();
                std::cout << "  DP[" << i << "][" << band_j << "] pos(" << i-1 << "," << j-1 << "): "
                          << "alice_token=" << token1 << ", bob_token=" << token2 
                          << ", match=" << match_result << ", delta=" << delta 
                          << ", diag=" << diag_val << ", up=" << up_val << ", left=" << left_val
                          << ", BEST=" << best_val << std::endl;
            }
        }
    }

    
    
    Integer final_score = dp[m][band_width-1];

    std::cout << "Final cell extraction: using dp[" << m << "][" << (band_width-1) << "] - EXACT same as FastEyeBLAST" << std::endl;
    
    
    std::cout << "=== DEBUG: BANDED DP FINAL SCORE (Party " << (party == ALICE ? "Alice" : "Bob") << ") ===" << std::endl;
    int revealed_score = final_score.reveal<int>();
    std::cout << "Final DP score before return: " << revealed_score << std::endl;
    
    
    std::cout << "=== DEBUG: LAST ROW OF DP MATRIX ===" << std::endl;
    for (int j = std::max(0, band_width-5); j < band_width; j++) {
        int cell_val = dp[m][j].reveal<int>();
        std::cout << "dp[" << m << "][" << j << "] = " << cell_val << std::endl;
    }
    
    std::cout << "EXACT original token-level banded Levenshtein DP completed.\n";

    return final_score;
}

void run_refine_gc(int        party,
                   const std::string& ip,
                   int        port,
                   const std::vector<uint64_t>& local_packed,
                   const std::vector<int>&      local_tokens,
                   int        bits_per_token,
                   int        tokens_per_word,
                   int        band_width,
                   const std::string&          output_file,
                   const std::string&          token_file) {

    std::cout << "Setting up EMP-Toolkit for party " << party << '\n';

    
    NetIO io(party == ALICE ? nullptr : ip.c_str(), port);
    setup_semi_honest(&io, party);
    std::cout << "Connected. Starting refinement circuit…\n";

    
    size_t local_packed_len  = local_packed.size();
    size_t remote_packed_len = 0;
    size_t local_token_len   = local_tokens.size();
    size_t remote_token_len  = 0;

    if (party == ALICE) {
        io.send_data(&local_packed_len, sizeof(size_t));
        io.recv_data(&remote_packed_len, sizeof(size_t));
        io.send_data(&local_token_len,   sizeof(size_t));
        io.recv_data(&remote_token_len,  sizeof(size_t));
    } else {
        io.recv_data(&remote_packed_len, sizeof(size_t));
        io.send_data(&local_packed_len,  sizeof(size_t));
        io.recv_data(&remote_token_len,  sizeof(size_t));
        io.send_data(&local_token_len,   sizeof(size_t));
    }

    const size_t max_packed_len = std::max(local_packed_len,  remote_packed_len);
    const size_t max_token_len  = std::max(local_token_len,   remote_token_len);

    
    std::vector<Integer> alice_words, bob_words;
    std::vector<Integer> alice_tokens, bob_tokens;
    alice_words .reserve(max_packed_len);
    bob_words   .reserve(max_packed_len);
    alice_tokens.reserve(max_token_len);
    bob_tokens  .reserve(max_token_len);

    
    for (size_t i = 0; i < max_packed_len; ++i) {
        uint64_t a = (party == ALICE && i < local_packed_len) ? local_packed[i] : 0ULL;
        uint64_t b = (party == BOB   && i < local_packed_len) ? local_packed[i] : 0ULL;
        alice_words.emplace_back(Integer(64, a, ALICE));
        bob_words  .emplace_back(Integer(64, b, BOB));
    }

    
    for (size_t i = 0; i < max_token_len; ++i) {
        int a = (party == ALICE && i < local_token_len) ? local_tokens[i] : 0;
        int b = (party == BOB   && i < local_token_len) ? local_tokens[i] : 0;
        alice_tokens.emplace_back(Integer(32, a, ALICE));
        bob_tokens  .emplace_back(Integer(32, b, BOB));
    }

    

    
    std::cout << "Phase 1: EXACT original seed extraction on packed words…\n";
    auto seeds_with_validity = SeedExtractionCircuit(
        alice_words, bob_words,
        bits_per_token,
        tokens_per_word
    );

    std::cout << "Phase 1b: EXACT original seed chaining…\n";
    auto chain = GreedyChainSeedsCircuit(
        seeds_with_validity,
          max_token_len,
          max_token_len
    );
    
    std::cout << "=== DEBUG: SEED CHAINING RESULT ===" << std::endl;
    std::cout << "Chain size: " << chain.size() << std::endl;
    if (chain.empty()) {
        std::cout << "WARNING: Chain is empty! This will cause early exit with negative score." << std::endl;
    }

    
    std::cout << "Phase 2: TOKEN-LEVEL banded dynamic programming - EXACT same as plain EyeBLAST…\n";

    if (party == ALICE) {
        std::cout << "=== PLAIN LIBRARY TEST ===\n";
        double plain_result = pp_eyeblast::normalized_similarity(local_tokens, local_tokens, band_width);
        std::cout << "Plain library result for identical sequences: " << plain_result << std::endl;
    }

    
    int alice_real_len = (party == ALICE) ? static_cast<int>(local_token_len)
                                          : static_cast<int>(remote_token_len);
    int bob_real_len   = (party == ALICE) ? static_cast<int>(remote_token_len)
                                          : static_cast<int>(local_token_len);

    Integer final_score = BandedDPCircuit(
        alice_tokens, bob_tokens,
        alice_real_len,     
        bob_real_len,       
        band_width,
        chain,
        party
    );

    
    std::cout << "Revealing final score…\n";
    int32_t raw_score = final_score.reveal<int32_t>();
    finalize_semi_honest();
    std::cout << "Finalizing protocol…\n";

    
    std::ofstream out(output_file, std::ios::trunc);
    if (out) {
        size_t alice_len = (party == ALICE) ? local_token_len : remote_token_len;
        size_t bob_len   = (party == ALICE) ? remote_token_len : local_token_len;
        out << raw_score << ' ' << alice_len << ' ' << bob_len << '\n';
    } else {
        std::cerr << "⚠️ Could not open " << output_file << '\n';
    }

    if (party == ALICE) {
        std::cout << "Refinement score (raw): " << raw_score << '\n';
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --role {alice|bob}     Party role (required)" << std::endl;
    std::cout << "  --ip IP_ADDRESS        IP address for Bob to connect to Alice" << std::endl;
    std::cout << "  --port PORT            Port number (default: 12346)" << std::endl;
    std::cout << "  --infile FILE          JSON file with tokens (required)" << std::endl;
    std::cout << "  --band_w WIDTH         Band width for DP (default: 5)" << std::endl;
    std::cout << "  --output FILE          Output file for result (default: refine_result.txt)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    
    int party = -1;
    std::string ip = "127.0.0.1";
    int port = 12346;
    std::string infile;
    int band_width = 5;
    std::string output_file = "refine_result.txt";
    
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--role" && i + 1 < argc) {
            std::string role = argv[++i];
            if (role == "alice") {
                party = ALICE;
            } else if (role == "bob") {
                party = BOB;
            } else {
                std::cerr << "Error: Role must be 'alice' or 'bob'" << std::endl;
                return 1;
            }
        } else if (arg == "--ip" && i + 1 < argc) {
            ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--infile" && i + 1 < argc) {
            infile = argv[++i];
        } else if (arg == "--band_w" && i + 1 < argc) {
            band_width = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    
    if (party == -1) {
        std::cerr << "Error: --role is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (infile.empty()) {
        std::cerr << "Error: --infile is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "=== Privacy-Preserving EyeBLAST Fine Refinement ===" << std::endl;
        std::cout << "Party: " << (party == ALICE ? "Alice" : "Bob") << std::endl;
        std::cout << "Input file: " << infile << std::endl;
        std::cout << "Band width: " << band_width << std::endl;
        std::cout << "Port: " << port << std::endl;
        
        
        std::cout << "Loading scanpath from " << infile << "..." << std::endl;
        
        std::ifstream file(infile);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << infile << std::endl;
            return 1;
        }
        
        nlohmann::json j;
        file >> j;
        
        std::vector<std::string> tokens = j.get<std::vector<std::string>>();
        file.close();
        
        if (tokens.empty()) {
            std::cerr << "Error: Empty token list" << std::endl;
            return 1;
        }
        
        std::cout << "Loaded " << tokens.size() << " tokens" << std::endl;
        
        
        std::cout << "Synchronizing token dictionaries..." << std::endl;
        
        
        std::set<std::string> comprehensive_vocab;
        
        
        const std::vector<char> prefixes = {'M', 'F', 'S'};
        for (char prefix : prefixes) {
            for (int bin = 0; bin < 8; ++bin) {
                comprehensive_vocab.insert(std::string(1, prefix) + std::to_string(bin));
            }
        }
        
        
        for (const std::string& token : tokens) {
            comprehensive_vocab.insert(get_global_instance().bin_duration(token));
        }
        
        
        for (const std::string& token : comprehensive_vocab) {
            get_global_instance().token_to_integer(token);
        }
        
        
        get_global_instance().freeze_dictionary();
        
        std::cout << "Token dictionary synchronized and frozen with " << comprehensive_vocab.size() << " tokens" << std::endl;
        
        
        std::cout << "Encoding and packing scanpath..." << std::endl;
        auto encoding = encode_and_pack(tokens);
        
        std::cout << "Generated " << encoding.packed.size() << " packed words" << std::endl;
        
        
        std::cout << "Starting garbled circuit protocol..." << std::endl;
        run_refine_gc(party, ip, port, encoding.packed, encoding.ids,encoding.bits_per_token, encoding.tokens_per_word, band_width, output_file, infile);
        
        std::cout << "Protocol completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}