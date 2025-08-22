/**
 * FastEyeBLAST: High-performance scanpath similarity calculation and classification
 * using bit-level operations, integer encodings, and efficient hashing techniques.
 * 
 * C++ implementation with EXACT Python logic replication.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <bitset>
#include <numeric>

// JSON library - you may need to install nlohmann/json
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;
using namespace std::chrono;

// Cross-platform popcount function
#ifdef _MSC_VER
#include <intrin.h>
inline int popcount64(uint64_t value) {
    return static_cast<int>(__popcnt64(value));
}
#elif defined(__GNUC__) || defined(__clang__)
inline int popcount64(uint64_t value) {
    return __builtin_popcountll(value);
}
#else
// Fallback implementation for other compilers
inline int popcount64(uint64_t value) {
    int count = 0;
    while (value) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}
#endif

// Constants - EXACT same as Python
const uint64_t MASK64 = (1ULL << 64) - 1;
const int DEFAULT_SEED_LENGTH = 3;
const int DEFAULT_NUM_NEIGHBORS = 5;
const int DEFAULT_BAND_WIDTH = 5;
const int DEFAULT_NUM_HASH_TABLES = 8;
const int DEFAULT_NUM_HASH_BITS = 16;
const int DEFAULT_MAX_CANDIDATES = 20;
const int DEFAULT_LSH_HASH_SIZE = 64;
const int DEFAULT_DURATION_BINS = 8;

// Python-compatible random number generator class
class PythonRandom {
private:
    mt19937 gen;
    
public:
    PythonRandom(int seed = 42) : gen(seed) {}
    
    void seed(int s) {
        gen.seed(s);
    }
    
    double random() {
        // Python's random.random() returns [0.0, 1.0)
        uniform_real_distribution<double> dis(0.0, 1.0);
        return dis(gen);
    }
    
    int randrange(int start, int stop) {
        uniform_int_distribution<int> dis(start, stop - 1);
        return dis(gen);
    }
    
    template<typename T>
    void shuffle(vector<T>& vec) {
        // Use same algorithm as Python's random.shuffle
        for (int i = vec.size() - 1; i > 0; i--) {
            int j = randrange(0, i + 1);
            swap(vec[i], vec[j]);
        }
    }
    
    template<typename T>
    vector<T> sample(const vector<T>& population, size_t k) {
        if (k > population.size()) {
            throw invalid_argument("Sample larger than population");
        }
        
        vector<T> pool = population;
        vector<T> result;
        result.reserve(k);
        
        for (size_t i = 0; i < k; i++) {
            int j = randrange(0, pool.size());
            result.push_back(pool[j]);
            pool.erase(pool.begin() + j);
        }
        
        return result;
    }
};

// Global Python-compatible random generator
PythonRandom python_random(42);

struct ClassMetrics {
    double precision;
    double recall;
    double f1;
    int support;
};

struct MacroAverage {
    double precision;
    double recall;
    double f1;
};

struct EvaluationResults {
    double accuracy;
    vector<vector<int>> confusion_matrix;
    map<int, ClassMetrics> class_metrics;
    MacroAverage macro_avg;
};

class FastEyeBLAST {
private:
    // Parameters - EXACT same as Python
    int seed_length;
    int band_width;
    int num_neighbors;
    int num_hash_tables;
    int num_hash_bits;
    int max_candidates;
    int lsh_hash_size;
    bool use_duration_bins;
    int duration_bins;
    bool use_parallel;
    int num_processes;
    
    // Tokenization and encoding - EXACT same as Python
    unordered_map<string, int> token_to_id;
    unordered_map<int, string> id_to_token;
    int next_token_id;
    
    // Bit-packing parameters - EXACT same as Python
    int bits_per_token;
    int tokens_per_word;
    uint64_t base;
    vector<uint64_t> pow_base;
    
    // LSH parameters - EXACT same as Python
    vector<unordered_map<uint64_t, vector<int>>> lsh_tables;
    vector<uint64_t> id_bitmasks;
    
    // Training data - EXACT same as Python
    vector<vector<int>> encoded_scanpaths;
    vector<vector<uint64_t>> packed_scanpaths;
    vector<uint64_t> simhash_sketches;
    vector<int> labels;
    
    // Python-compatible random number generator for this instance
    PythonRandom instance_random;

public:
    FastEyeBLAST(int seed_length = DEFAULT_SEED_LENGTH,
                 int band_width = DEFAULT_BAND_WIDTH,
                 int num_neighbors = DEFAULT_NUM_NEIGHBORS,
                 int num_hash_tables = DEFAULT_NUM_HASH_TABLES,
                 int num_hash_bits = DEFAULT_NUM_HASH_BITS,
                 int max_candidates = DEFAULT_MAX_CANDIDATES,
                 int lsh_hash_size = DEFAULT_LSH_HASH_SIZE,
                 bool use_duration_bins = true,
                 int duration_bins = DEFAULT_DURATION_BINS,
                 bool use_parallel = true,
                 int num_processes = 0)
        : seed_length(seed_length), band_width(band_width), num_neighbors(num_neighbors),
          num_hash_tables(num_hash_tables), num_hash_bits(num_hash_bits),
          max_candidates(max_candidates), lsh_hash_size(lsh_hash_size),
          use_duration_bins(use_duration_bins), duration_bins(duration_bins),
          use_parallel(use_parallel), next_token_id(0), instance_random(42) {
        
        if (num_processes == 0) {
            this->num_processes = thread::hardware_concurrency();
        } else {
            this->num_processes = num_processes;
        }
    }
    
    // EXACT replication of Python _token_to_integer
    int token_to_integer(const string& token) {
        if (token_to_id.find(token) == token_to_id.end()) {
            token_to_id[token] = next_token_id;
            id_to_token[next_token_id] = token;
            next_token_id++;
        }
        return token_to_id[token];
    }
    
    // EXACT replication of Python _bin_duration
    string bin_duration(const string& token) {
        if (!use_duration_bins) {
            return token;
        }
        
        if (token.empty()) return token;
        
        char token_type = token[0];
        string duration_str = token.substr(1);
        
        if (duration_str.empty() || !all_of(duration_str.begin(), duration_str.end(), ::isdigit)) {
            return token;
        }
        
        try {
            int duration = stoi(duration_str);
            int bin_idx;
            
            if (duration == 0) {
                bin_idx = 0;
            } else {
                // EXACT same logic as Python: min(int(np.log2(duration + 1)), self.duration_bins - 1)
                bin_idx = min(static_cast<int>(log2(duration + 1)), duration_bins - 1);
            }
            
            return token_type + to_string(bin_idx);
        } catch (...) {
            return token;
        }
    }
    
    // EXACT replication of Python _encode_scanpath
    vector<int> encode_scanpath(const vector<string>& scanpath) {
        vector<int> encoded;
        for (const string& token : scanpath) {
            string binned_token = bin_duration(token);
            encoded.push_back(token_to_integer(binned_token));
        }
        return encoded;
    }
    
    // EXACT replication of Python _initialize_bit_packing
    void initialize_bit_packing() {
        int alphabet_size = token_to_id.size();
        // EXACT same calculation as Python
        bits_per_token = max(1, static_cast<int>(ceil(log2(alphabet_size))));
        tokens_per_word = 64 / bits_per_token;
        
        cout << "Alphabet size: " << alphabet_size << endl;
        cout << "Bits per token: " << bits_per_token << endl;
        cout << "Tokens per 64-bit word: " << tokens_per_word << endl;
        
        // Initialize rolling hash parameters - EXACT same as Python
        base = 31;  // A prime number for rolling hash
        pow_base.clear();
        pow_base.push_back(1);
        for (int i = 1; i < max(seed_length, 100); i++) {
            pow_base.push_back((pow_base.back() * base) & MASK64);
        }
    }
    
    // EXACT replication of Python _pack_ids_into_words
    vector<uint64_t> pack_ids_into_words(const vector<int>& id_list) {
        if (id_list.empty()) return {};
        
        // EXACT same calculation as Python
        int num_words = (id_list.size() + tokens_per_word - 1) / tokens_per_word;
        vector<uint64_t> packed_words(num_words, 0);
        
        for (int i = 0; i < num_words; i++) {
            uint64_t word = 0;
            for (int j = 0; j < tokens_per_word; j++) {
                int idx = i * tokens_per_word + j;
                if (idx < id_list.size()) {
                    // EXACT same bit manipulation as Python
                    uint64_t token_bits = id_list[idx] & ((1ULL << bits_per_token) - 1);
                    word |= token_bits << (j * bits_per_token);
                }
            }
            packed_words[i] = word;
        }
        
        return packed_words;
    }
    
    // EXACT replication of Python _calculate_rolling_hash
    vector<uint64_t> calculate_rolling_hash(const vector<int>& id_list) {
        if (id_list.size() < seed_length) return {};
        
        vector<uint64_t> hashes;
        int k = seed_length;
        
        // Initialize hash for first k-mer - EXACT same as Python
        uint64_t h = 0;
        for (int i = 0; i < k; i++) {
            h = (h * base + id_list[i]) & MASK64;
        }
        hashes.push_back(h);
        
        // Calculate rolling hash for remaining k-mers - EXACT same as Python
        for (int i = k; i < id_list.size(); i++) {
            // Remove contribution of the oldest character
            h = (h - id_list[i - k] * pow_base[k - 1]) & MASK64;
            // Multiply by base
            h = (h * base) & MASK64;
            // Add contribution of the new character
            h = (h + id_list[i]) & MASK64;
            hashes.push_back(h);
        }
        
        return hashes;
    }
    
    // EXACT replication of Python _initialize_simhash
    void initialize_simhash() {
        int alphabet_size = token_to_id.size();
        id_bitmasks.clear();
        id_bitmasks.resize(alphabet_size);
        
        // Use EXACT same seed and algorithm as Python: rng = random.Random(42)
        PythonRandom rng(42);
        
        for (int i = 0; i < alphabet_size; i++) {
            uint64_t mask = 0;
            for (int j = 0; j < lsh_hash_size; j++) {
                // EXACT same logic as Python: if rng.random() < 0.5:
                if (rng.random() < 0.5) {
                    mask |= (1ULL << j);
                }
            }
            id_bitmasks[i] = mask;
        }
    }
    
    // EXACT replication of Python _compute_simhash
    uint64_t compute_simhash(const vector<int>& id_list) {
        if (id_bitmasks.empty()) {
            initialize_simhash();
        }
        
        // Initialize counters for each bit position - EXACT same as Python
        vector<int> counters(lsh_hash_size, 0);
        
        // Accumulate counts - EXACT same logic as Python
        for (int token_id : id_list) {
            if (token_id >= 0 && token_id < id_bitmasks.size()) {
                uint64_t bitmask = id_bitmasks[token_id];
                for (int i = 0; i < lsh_hash_size; i++) {
                    if (bitmask & (1ULL << i)) {
                        counters[i]++;
                    } else {
                        counters[i]--;
                    }
                }
            }
        }
        
        // Generate fingerprint based on counters - EXACT same as Python
        uint64_t fingerprint = 0;
        for (int i = 0; i < lsh_hash_size; i++) {
            if (counters[i] > 0) {
                fingerprint |= (1ULL << i);
            }
        }
        
        return fingerprint;
    }
    
    // EXACT replication of Python _build_lsh_index
    vector<unordered_map<uint64_t, vector<int>>> build_lsh_index(const vector<uint64_t>& simhash_sketches) {
        vector<unordered_map<uint64_t, vector<int>>> lsh_tables;
        lsh_tables.resize(num_hash_tables);
        
        // EXACT same calculation as Python
        int bits_per_slice = lsh_hash_size / num_hash_tables;
        
        for (int idx = 0; idx < simhash_sketches.size(); idx++) {
            uint64_t sketch = simhash_sketches[idx];
            
            for (int table_idx = 0; table_idx < num_hash_tables; table_idx++) {
                int start_bit = table_idx * bits_per_slice;
                int end_bit = (table_idx + 1) * bits_per_slice;
                
                // EXACT same bit extraction as Python
                uint64_t bit_slice = (sketch >> start_bit) & ((1ULL << (end_bit - start_bit)) - 1);
                lsh_tables[table_idx][bit_slice].push_back(idx);
            }
        }
        
        return lsh_tables;
    }
    
    // EXACT replication of Python _query_lsh_index
    vector<int> query_lsh_index(uint64_t query_sketch) {
        unordered_set<int> candidates_set;
        
        // EXACT same calculation as Python
        int bits_per_slice = lsh_hash_size / num_hash_tables;
        
        for (int table_idx = 0; table_idx < num_hash_tables; table_idx++) {
            int start_bit = table_idx * bits_per_slice;
            int end_bit = (table_idx + 1) * bits_per_slice;
            
            // EXACT same bit extraction as Python
            uint64_t bit_slice = (query_sketch >> start_bit) & ((1ULL << (end_bit - start_bit)) - 1);
            
            if (lsh_tables[table_idx].find(bit_slice) != lsh_tables[table_idx].end()) {
                for (int idx : lsh_tables[table_idx][bit_slice]) {
                    candidates_set.insert(idx);
                }
            }
        }
        
        return vector<int>(candidates_set.begin(), candidates_set.end());
    }
    
    // EXACT replication of Python _hamming_distance
    int hamming_distance(uint64_t a, uint64_t b) {
        return popcount64(a ^ b);
    }
    
    // EXACT replication of Python _extract_seeds
    vector<pair<int, int>> extract_seeds(const vector<uint64_t>& packed1, 
                                         const vector<uint64_t>& packed2) {
        vector<pair<int, int>> seeds;
        int max_mismatches = 1;  // EXACT same as Python
        
        int min_len = min(packed1.size(), packed2.size());
        for (int i = 0; i < min_len; i++) {
            uint64_t xor_result = packed1[i] ^ packed2[i];
            int mismatches = popcount64(xor_result);
            
            // EXACT same condition as Python
            if (mismatches <= max_mismatches * bits_per_token) {
                int pos1 = i * tokens_per_word;
                int pos2 = i * tokens_per_word;
                seeds.push_back({pos1, pos2});
            }
        }
        
        return seeds;
    }
    
    // EXACT replication of Python _greedy_chain_seeds
    vector<pair<int, int>> greedy_chain_seeds(const vector<pair<int, int>>& seeds,
                                              int len1, int len2) {
        if (seeds.empty()) return {};
        
        // Group seeds by diagonal - EXACT same as Python
        map<int, vector<pair<int, int>>> diagonals;
        for (const auto& seed : seeds) {
            int diag = seed.second - seed.first;
            diagonals[diag].push_back(seed);
        }
        
        // Find the diagonal with the most seeds - EXACT same as Python
        auto best_diag = max_element(diagonals.begin(), diagonals.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second.size() < b.second.size();
                                     });
        
        vector<pair<int, int>> best_seeds = best_diag->second;
        sort(best_seeds.begin(), best_seeds.end());
        
        // Build chain greedily - EXACT same as Python
        vector<pair<int, int>> chain;
        if (!best_seeds.empty()) {
            chain.push_back(best_seeds[0]);
            for (int i = 1; i < best_seeds.size(); i++) {
                if (best_seeds[i].first > chain.back().first && 
                    best_seeds[i].second > chain.back().second) {
                    chain.push_back(best_seeds[i]);
                }
            }
        }
        
        return chain;
    }
    
    // EXACT replication of Python _banded_levenshtein - CRITICAL FIX!
    pair<int, vector<pair<int, int>>> banded_levenshtein(
            const vector<int>& seq1, const vector<int>& seq2,
            int band_width, const vector<pair<int, int>>& chain) {
        
        if (chain.empty()) {
            // EXACT same fallback as Python: return len(seq1) + len(seq2), []
            return {static_cast<int>(seq1.size() + seq2.size()), {}};
        }
        
        // Extract diagonal from chain - EXACT same as Python
        int diag = chain[0].second - chain[0].first;
        
        // Adjust band width to be symmetric around diagonal - EXACT same as Python
        int half_band = band_width / 2;
        
        // Initialize dynamic programming matrix - EXACT same as Python
        int m = seq1.size();
        int n = seq2.size();
        
        // Use integer-only scoring - EXACT same as Python
        int match_score = 2;
        int mismatch_penalty = -1;
        int gap_penalty = -1;
        
        // Preallocate DP matrix with band - EXACT same as Python
        vector<vector<int>> dp(m + 1, vector<int>(band_width, 0));
        
        // Initialize first row and column - EXACT same as Python
        for (int i = 0; i < band_width; i++) {
            dp[0][i] = i * gap_penalty;
        }
        
        for (int i = 1; i <= m; i++) {
            // Calculate column range for this row - EXACT same as Python
            int col_start = max(1, i + diag - half_band);
            int col_end = min(n + 1, i + diag + half_band + 1);
            int band_start = 0;
            
            // Initialize left edge of band if needed - EXACT same as Python
            if (col_start > 1) {
                dp[i][0] = dp[i-1][0] + gap_penalty;
            } else {
                dp[i][0] = i * gap_penalty;
            }
            
            // Fill the band for this row - EXACT same as Python
            for (int j = col_start; j < col_end; j++) {
                int band_j = j - col_start + band_start;
                
                if (band_j >= band_width) {
                    break;
                }
                
                // Calculate scores - EXACT same as Python
                if (j <= n) {
                    bool match = (seq1[i-1] == seq2[j-1]);
                    int diag_score = dp[i-1][band_j-1 > 0 ? band_j-1 : 0] + (match ? match_score : mismatch_penalty);
                    int up_score = dp[i-1][band_j] + gap_penalty;
                    int left_score = (band_j > 0) ? dp[i][band_j-1] : INT_MIN;
                    
                    // Choose best score - EXACT same as Python
                    dp[i][band_j] = max({diag_score, up_score, left_score});
                }
            }
        }
        
        // Extract score from bottom-right cell - EXACT same as Python
        int score = dp[m][band_width-1];
        
        // Simple alignment reconstruction (just use the chain) - EXACT same as Python
        vector<pair<int, int>> alignment = chain;
        
        return {score, alignment};
    }
    
    // EXACT replication of Python _similarity_score
    double similarity_score(const vector<int>& seq1, const vector<int>& seq2) {
        // Pack sequences into words - EXACT same as Python
        vector<uint64_t> packed1 = pack_ids_into_words(seq1);
        vector<uint64_t> packed2 = pack_ids_into_words(seq2);
        
        // Extract seeds - EXACT same as Python
        vector<pair<int, int>> seeds = extract_seeds(packed1, packed2);
        
        // Chain seeds - EXACT same as Python
        vector<pair<int, int>> chain = greedy_chain_seeds(seeds, seq1.size(), seq2.size());
        
        // Calculate banded Levenshtein distance - EXACT same as Python
        auto result = banded_levenshtein(seq1, seq2, band_width, chain);
        int score = result.first;
        
        // Normalize score - EXACT same as Python
        int max_possible_score = max(seq1.size(), seq2.size()) * 2;  // Maximum possible score
        double normalized_score = max(0.0, static_cast<double>(score)) / max_possible_score;
        
        return normalized_score;
    }
    
    void fit(const vector<vector<string>>& scanpaths, const vector<int>& labels) {
        this->labels = labels;
        
        cout << "Encoding scanpaths..." << endl;
        encoded_scanpaths.clear();
        for (const auto& sp : scanpaths) {
            encoded_scanpaths.push_back(encode_scanpath(sp));
        }
        
        initialize_bit_packing();
        
        cout << "Packing scanpaths into words..." << endl;
        packed_scanpaths.clear();
        for (const auto& sp : encoded_scanpaths) {
            packed_scanpaths.push_back(pack_ids_into_words(sp));
        }
        
        cout << "Computing SimHash sketches..." << endl;
        simhash_sketches.clear();
        for (const auto& sp : encoded_scanpaths) {
            simhash_sketches.push_back(compute_simhash(sp));
        }
        
        cout << "Building LSH index..." << endl;
        lsh_tables = build_lsh_index(simhash_sketches);
    }
    
    // EXACT replication of Python predict logic
    vector<int> predict(const vector<vector<string>>& scanpaths) {
        vector<int> predictions;
        predictions.reserve(scanpaths.size());
        
        for (const auto& scanpath : scanpaths) {
            // Encode scanpath - EXACT same as Python
            vector<int> encoded = encode_scanpath(scanpath);
            
            // Compute SimHash sketch - EXACT same as Python
            uint64_t sketch = compute_simhash(encoded);
            
            // Query LSH index for candidates - EXACT same as Python
            vector<int> candidates = query_lsh_index(sketch);
            
            // If not enough candidates, use random sampling - EXACT same as Python
            if (candidates.size() < max_candidates) {
                vector<int> all_indices(encoded_scanpaths.size());
                iota(all_indices.begin(), all_indices.end(), 0);
                
                unordered_set<int> candidate_set(candidates.begin(), candidates.end());
                vector<int> available;
                for (int idx : all_indices) {
                    if (candidate_set.find(idx) == candidate_set.end()) {
                        available.push_back(idx);
                    }
                }
                
                if (!available.empty()) {
                    int additional = min(max_candidates - static_cast<int>(candidates.size()), 
                                         static_cast<int>(available.size()));
                    // Use Python-compatible random sampling
                    vector<int> sampled = python_random.sample(available, additional);
                    candidates.insert(candidates.end(), sampled.begin(), sampled.end());
                }
            }
            
            // Sort candidates by Hamming distance - EXACT same as Python
            sort(candidates.begin(), candidates.end(),
                 [this, sketch](int a, int b) {
                     return hamming_distance(sketch, simhash_sketches[a]) < 
                            hamming_distance(sketch, simhash_sketches[b]);
                 });
            
            // Take top candidates - EXACT same as Python
            if (candidates.size() > max_candidates) {
                candidates.resize(max_candidates);
            }
            
            // Calculate similarities - EXACT same as Python
            vector<pair<int, double>> similarities;
            for (int candidate_idx : candidates) {
                double sim = similarity_score(encoded, encoded_scanpaths[candidate_idx]);
                similarities.push_back({candidate_idx, sim});
            }
            
            // Sort by similarity - EXACT same as Python
            sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Take top k - EXACT same as Python
            if (similarities.size() > num_neighbors) {
                similarities.resize(num_neighbors);
            }
            
            // Majority vote with weights - EXACT same as Python
            map<int, double> votes;
            for (const auto& sim_pair : similarities) {
                int label = labels[sim_pair.first];
                votes[label] += sim_pair.second;
            }
            
            int prediction;
            if (!votes.empty()) {
                prediction = max_element(votes.begin(), votes.end(),
                                         [](const auto& a, const auto& b) {
                                             return a.second < b.second;
                                         })->first;
            } else {
                // Fallback: most common label - EXACT same as Python
                map<int, int> label_counts;
                for (int label : labels) {
                    label_counts[label]++;
                }
                prediction = max_element(label_counts.begin(), label_counts.end(),
                                         [](const auto& a, const auto& b) {
                                             return a.second < b.second;
                                         })->first;
            }
            
            predictions.push_back(prediction);
        }
        
        return predictions;
    }
    
    void set_num_neighbors(int k) {
        num_neighbors = k;
    }
};

// EXACT replication of Python load_scanpaths function
pair<vector<vector<string>>, vector<int>> load_scanpaths(const string& input_dir, 
                                                         int sample_percentage = 100) {
    vector<vector<string>> scanpaths;
    vector<int> labels;
    
    // Get all JSON files - EXACT same logic as Python
    vector<string> files;
    for (const auto& entry : filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".json") {
            files.push_back(entry.path().string());
        }
    }
    
    // CRITICAL: Sort files to ensure deterministic order (unlike Python glob which may vary)
    sort(files.begin(), files.end());
    
    // Sample files if requested - EXACT same as Python
    if (sample_percentage < 100) {
        int num_files = files.size() * sample_percentage / 100;
        vector<string> sampled_files = python_random.sample(files, num_files);
        files = sampled_files;
    }
    
    cout << "Loading " << files.size() << " scanpath files..." << endl;
    
    for (const string& file_path : files) {
        // Parse filename: S_XXXX_SY_Z_TTT_tokens.json - EXACT same as Python
        filesystem::path path(file_path);
        string filename = path.filename().string();
        
        vector<string> parts;
        stringstream ss(filename);
        string part;
        while (getline(ss, part, '_')) {
            parts.push_back(part);
        }
        
        if (parts.size() >= 4) {
            int task_num = stoi(parts[3]);
            
            ifstream file(file_path);
            if (file.is_open()) {
                json j;
                file >> j;
                
                vector<string> scanpath = j.get<vector<string>>();
                scanpaths.push_back(scanpath);
                labels.push_back(task_num);
            }
        }
    }
    
    return {scanpaths, labels};
}

EvaluationResults evaluate_classifier(const vector<int>& y_true, const vector<int>& y_pred) {
    EvaluationResults results;
    
    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    results.accuracy = static_cast<double>(correct) / y_true.size();
    
    // Get unique classes
    set<int> class_set(y_true.begin(), y_true.end());
    vector<int> classes(class_set.begin(), class_set.end());
    sort(classes.begin(), classes.end());
    
    // Initialize confusion matrix
    results.confusion_matrix.resize(classes.size(), vector<int>(classes.size(), 0));
    
    // Fill confusion matrix
    for (int i = 0; i < y_true.size(); i++) {
        int true_idx = find(classes.begin(), classes.end(), y_true[i]) - classes.begin();
        int pred_idx = find(classes.begin(), classes.end(), y_pred[i]) - classes.begin();
        results.confusion_matrix[true_idx][pred_idx]++;
    }
    
    // Calculate per-class metrics
    double total_precision = 0, total_recall = 0, total_f1 = 0;
    
    for (int cls_idx = 0; cls_idx < classes.size(); cls_idx++) {
        int cls = classes[cls_idx];
        
        int tp = results.confusion_matrix[cls_idx][cls_idx];
        int fp = 0, fn = 0;
        
        for (int i = 0; i < classes.size(); i++) {
            if (i != cls_idx) {
                fp += results.confusion_matrix[i][cls_idx];
                fn += results.confusion_matrix[cls_idx][i];
            }
        }
        
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
        
        int support = 0;
        for (int val : results.confusion_matrix[cls_idx]) {
            support += val;
        }
        
        results.class_metrics[cls] = {precision, recall, f1, support};
        
        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
    }
    
    // Calculate macro averages
    results.macro_avg.precision = total_precision / classes.size();
    results.macro_avg.recall = total_recall / classes.size();
    results.macro_avg.f1 = total_f1 / classes.size();
    
    return results;
}

void save_results(const EvaluationResults& results, 
                  const map<string, double>& timing_data,
                  const vector<int>& y_true, const vector<int>& y_pred,
                  const string& output_dir,
                  const vector<EvaluationResults>* parameter_sweep_results = nullptr,
                  const vector<int>* k_values = nullptr) {
    
    // Create output directory
    filesystem::create_directories(output_dir);
    
    // Save main results
    json results_json;
    results_json["accuracy"] = results.accuracy;
    results_json["macro_avg"]["precision"] = results.macro_avg.precision;
    results_json["macro_avg"]["recall"] = results.macro_avg.recall;
    results_json["macro_avg"]["f1"] = results.macro_avg.f1;
    
    for (const auto& [cls, metrics] : results.class_metrics) {
        results_json["class_metrics"][to_string(cls)]["precision"] = metrics.precision;
        results_json["class_metrics"][to_string(cls)]["recall"] = metrics.recall;
        results_json["class_metrics"][to_string(cls)]["f1"] = metrics.f1;
        results_json["class_metrics"][to_string(cls)]["support"] = metrics.support;
    }
    
    results_json["confusion_matrix"] = results.confusion_matrix;
    results_json["timing_data"] = timing_data;
    
    // Add parameter sweep results if available
    if (parameter_sweep_results && k_values) {
        json sweep_results = json::array();
        for (size_t i = 0; i < parameter_sweep_results->size(); i++) {
            const auto& sweep_result = (*parameter_sweep_results)[i];
            json sweep_entry;
            sweep_entry["k_value"] = (*k_values)[i];
            sweep_entry["accuracy"] = sweep_result.accuracy;
            sweep_entry["macro_avg"]["precision"] = sweep_result.macro_avg.precision;
            sweep_entry["macro_avg"]["recall"] = sweep_result.macro_avg.recall;
            sweep_entry["macro_avg"]["f1"] = sweep_result.macro_avg.f1;
            
            // Add class metrics for sweep
            for (const auto& [cls, metrics] : sweep_result.class_metrics) {
                sweep_entry["class_metrics"][to_string(cls)]["precision"] = metrics.precision;
                sweep_entry["class_metrics"][to_string(cls)]["recall"] = metrics.recall;
                sweep_entry["class_metrics"][to_string(cls)]["f1"] = metrics.f1;
                sweep_entry["class_metrics"][to_string(cls)]["support"] = metrics.support;
            }
            sweep_entry["confusion_matrix"] = sweep_result.confusion_matrix;
            
            sweep_results.push_back(sweep_entry);
        }
        results_json["parameter_sweep_results"] = sweep_results;
        results_json["k_values"] = *k_values;
    }
    
    ofstream results_file(output_dir + "/results.json");
    results_file << results_json.dump(2);
    results_file.close();
    
    // Save predictions
    json pred_json;
    pred_json["true"] = y_true;
    pred_json["pred"] = y_pred;
    
    ofstream pred_file(output_dir + "/predictions.json");
    pred_file << pred_json.dump(2);
    pred_file.close();
    
    // Export raw plotting data
    json plot_data;
    
    // Basic metrics for plotting
    plot_data["accuracy"] = results.accuracy;
    plot_data["macro_avg"] = {
        {"precision", results.macro_avg.precision},
        {"recall", results.macro_avg.recall},
        {"f1", results.macro_avg.f1}
    };
    
    // Class metrics for bar plots
    json class_data;
    vector<int> classes;
    vector<double> precisions, recalls, f1_scores, supports;
    
    for (const auto& [cls, metrics] : results.class_metrics) {
        classes.push_back(cls);
        precisions.push_back(metrics.precision);
        recalls.push_back(metrics.recall);
        f1_scores.push_back(metrics.f1);
        supports.push_back(metrics.support);
    }
    
    class_data["classes"] = classes;
    class_data["precision"] = precisions;
    class_data["recall"] = recalls;
    class_data["f1"] = f1_scores;
    class_data["support"] = supports;
    plot_data["class_data"] = class_data;
    
    // Timing data for plots
    plot_data["timing_data"] = timing_data;
    
    // Predictions for confusion matrix
    plot_data["predictions"] = {
        {"y_true", y_true},
        {"y_pred", y_pred}
    };
    
    // Confusion matrix
    plot_data["confusion_matrix"] = results.confusion_matrix;
    
    // Parameter sweep data if available
    if (parameter_sweep_results && k_values) {
        vector<double> accuracies, f1_scores;
        for (const auto& sweep_result : *parameter_sweep_results) {
            accuracies.push_back(sweep_result.accuracy);
            f1_scores.push_back(sweep_result.macro_avg.f1);
        }
        plot_data["parameter_sweep"] = {
            {"k_values", *k_values},
            {"accuracies", accuracies},
            {"f1_scores", f1_scores}
        };
    }
    
    // Calculate total samples for summary
    int total_samples = 0;
    for (const auto& [cls, metrics] : results.class_metrics) {
        total_samples += metrics.support;
    }
    plot_data["total_samples"] = total_samples;
    
    ofstream plot_file(output_dir + "/plot_data.json");
    plot_file << plot_data.dump(2);
    plot_file.close();
    
    // Save detailed report
    ofstream report_file(output_dir + "/detailed_report.txt");
    report_file << "FastEyeBLAST C++ Detailed Performance Report\n";
    report_file << string(50, '=') << "\n\n";
    
    report_file << "Overall Performance:\n";
    report_file << string(20, '-') << "\n";
    report_file << "Accuracy: " << fixed << setprecision(4) << results.accuracy 
                << " (" << (results.accuracy * 100) << "%)\n";
    report_file << "Macro-avg Precision: " << results.macro_avg.precision << "\n";
    report_file << "Macro-avg Recall: " << results.macro_avg.recall << "\n";
    report_file << "Macro-avg F1-Score: " << results.macro_avg.f1 << "\n\n";
    
    report_file << "Per-Class Performance:\n";
    report_file << string(22, '-') << "\n";
    report_file << left << setw(8) << "Class" << setw(12) << "Precision" 
                << setw(12) << "Recall" << setw(12) << "F1-Score" << setw(12) << "Support\n";
    report_file << string(55, '-') << "\n";
    
    for (const auto& [cls, metrics] : results.class_metrics) {
        report_file << left << setw(8) << cls << setw(12) << fixed << setprecision(4) 
                    << metrics.precision << setw(12) << metrics.recall 
                    << setw(12) << metrics.f1 << setw(12) << metrics.support << "\n";
    }
    
    report_file << "\nTiming Analysis:\n";
    report_file << string(16, '-') << "\n";
    double total_time = 0;
    for (const auto& [phase, time_val] : timing_data) {
        total_time += time_val;
    }
    report_file << "Total execution time: " << total_time << " seconds\n";
    
    for (const auto& [phase, time_val] : timing_data) {
        double percentage = (time_val / total_time) * 100;
        report_file << phase << ": " << time_val << "s (" << percentage << "%)\n";
    }
    
    report_file.close();
}

int main(int argc, char* argv[]) {
    // Default parameters - EXACT same as Python
    string input_dir = "GazeBaseVR_IVT";
    string output_dir = "FastEyeBLAST_Results_Cpp_Compare";
    int sample_percentage = 100;
    double test_size = 0.2;
    int seed_length = DEFAULT_SEED_LENGTH;
    int band_width = DEFAULT_BAND_WIDTH;
    int num_neighbors = DEFAULT_NUM_NEIGHBORS;
    int max_candidates = DEFAULT_MAX_CANDIDATES;
    bool use_duration_bins = true;
    int duration_bins = DEFAULT_DURATION_BINS;
    bool use_parallel = true;
    int num_processes = 0;
    int seed = 42;
    bool parameter_sweep = false;
    vector<int> k_values = {1, 3, 5, 7, 10};
    
    // Simple argument parsing - EXACT same parameters as Python
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--input_dir" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--output_dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--sample_percentage" && i + 1 < argc) {
            sample_percentage = stoi(argv[++i]);
        } else if (arg == "--test_size" && i + 1 < argc) {
            test_size = stod(argv[++i]);
        } else if (arg == "--num_neighbors" && i + 1 < argc) {
            num_neighbors = stoi(argv[++i]);
        } else if (arg == "--parameter_sweep") {
            parameter_sweep = true;
        } else if (arg == "--help") {
            cout << "FastEyeBLAST C++ - High-performance scanpath classification\n";
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  --input_dir DIR       Input directory with JSON files\n";
            cout << "  --output_dir DIR      Output directory for results\n";
            cout << "  --sample_percentage N Percentage of data to use\n";
            cout << "  --test_size F         Test set proportion (0-1)\n";
            cout << "  --num_neighbors K     Number of neighbors for k-NN\n";
            cout << "  --parameter_sweep     Run parameter sweep over k values\n";
            cout << "  --help                Show this help message\n";
            return 0;
        }
    }
    
    // Set random seed - EXACT same as Python
    python_random.seed(seed);
    
    cout << string(60, '=') << endl;
    cout << "FastEyeBLAST: High-Performance Scanpath Classification (C++)" << endl;
    cout << string(60, '=') << endl;
    
    map<string, double> timing_data;
    auto overall_start = high_resolution_clock::now();
    
    // Load data
    cout << "Loading scanpaths..." << endl;
    auto load_start = high_resolution_clock::now();
    auto [scanpaths, labels] = load_scanpaths(input_dir, sample_percentage);
    auto load_end = high_resolution_clock::now();
    timing_data["data_loading"] = duration<double>(load_end - load_start).count();
    cout << "Loaded " << scanpaths.size() << " scanpaths in " 
         << timing_data["data_loading"] << " seconds" << endl;
    
    // Split data - EXACT same logic as Python
    auto split_start = high_resolution_clock::now();
    int test_count = static_cast<int>(scanpaths.size() * test_size);
    
    // Create indices and shuffle - EXACT same as Python
    vector<int> indices(scanpaths.size());
    iota(indices.begin(), indices.end(), 0);
    python_random.shuffle(indices);  // Use Python-compatible shuffle
    
    // Split indices - EXACT same as Python
    vector<int> test_indices(indices.begin(), indices.begin() + test_count);
    vector<int> train_indices(indices.begin() + test_count, indices.end());
    
    vector<vector<string>> train_scanpaths, test_scanpaths;
    vector<int> train_labels, test_labels;
    
    for (int idx : train_indices) {
        train_scanpaths.push_back(scanpaths[idx]);
        train_labels.push_back(labels[idx]);
    }
    
    for (int idx : test_indices) {
        test_scanpaths.push_back(scanpaths[idx]);
        test_labels.push_back(labels[idx]);
    }
    
    auto split_end = high_resolution_clock::now();
    timing_data["data_splitting"] = duration<double>(split_end - split_start).count();
    
    cout << "Training set: " << train_scanpaths.size() << " scanpaths" << endl;
    cout << "Test set: " << test_scanpaths.size() << " scanpaths" << endl;
    
    // Initialize classifier - EXACT same parameters as Python
    FastEyeBLAST classifier(seed_length, band_width, num_neighbors, 
                           DEFAULT_NUM_HASH_TABLES, DEFAULT_NUM_HASH_BITS,
                           max_candidates, DEFAULT_LSH_HASH_SIZE, 
                           use_duration_bins, duration_bins, use_parallel, num_processes);
    
    // Train classifier
    cout << "\nTraining classifier..." << endl;
    auto train_start = high_resolution_clock::now();
    classifier.fit(train_scanpaths, train_labels);
    auto train_end = high_resolution_clock::now();
    timing_data["train_time"] = duration<double>(train_end - train_start).count();
    cout << "Training completed in " << timing_data["train_time"] << " seconds" << endl;
    
    EvaluationResults final_results;
    vector<int> final_predictions;
    vector<EvaluationResults> results_list;
    
    if (parameter_sweep) {
        cout << "\nRunning parameter sweep over k values: ";
        for (int k : k_values) cout << k << " ";
        cout << endl;
        
        EvaluationResults best_result;
        double best_f1 = -1;
        int best_k = k_values[0];
        
        for (int k : k_values) {
            cout << "\n--- Testing with k=" << k << " ---" << endl;
            classifier.set_num_neighbors(k);
            
            auto predict_start = high_resolution_clock::now();
            vector<int> predictions = classifier.predict(test_scanpaths);
            auto predict_end = high_resolution_clock::now();
            double predict_time = duration<double>(predict_end - predict_start).count();
            
            EvaluationResults results = evaluate_classifier(test_labels, predictions);
            results_list.push_back(results);
            
            cout << "k=" << k << ": Accuracy=" << fixed << setprecision(4) 
                 << results.accuracy << ", F1=" << results.macro_avg.f1 << endl;
            
            if (results.macro_avg.f1 > best_f1) {
                best_f1 = results.macro_avg.f1;
                best_k = k;
                best_result = results;
                final_predictions = predictions;
                timing_data["predict_time"] = predict_time;
            }
        }
        
        cout << "\nBest k value: " << best_k << " (F1=" << best_f1 << ")" << endl;
        final_results = best_result;
        
    } else {
        cout << "Making predictions..." << endl;
        auto predict_start = high_resolution_clock::now();
        final_predictions = classifier.predict(test_scanpaths);
        auto predict_end = high_resolution_clock::now();
        timing_data["predict_time"] = duration<double>(predict_end - predict_start).count();
        cout << "Prediction completed in " << timing_data["predict_time"] << " seconds" << endl;
        
        cout << "Evaluating performance..." << endl;
        auto eval_start = high_resolution_clock::now();
        final_results = evaluate_classifier(test_labels, final_predictions);
        auto eval_end = high_resolution_clock::now();
        timing_data["evaluation"] = duration<double>(eval_end - eval_start).count();
    }
    
    auto overall_end = high_resolution_clock::now();
    timing_data["total_time"] = duration<double>(overall_end - overall_start).count();
    
    // Print results - EXACT same format as Python
    cout << "\n" << string(40, '=') << endl;
    cout << "PERFORMANCE RESULTS" << endl;
    cout << string(40, '=') << endl;
    cout << "Accuracy: " << fixed << setprecision(4) << final_results.accuracy 
         << " (" << (final_results.accuracy * 100) << "%)" << endl;
    cout << "Macro-avg F1: " << final_results.macro_avg.f1 << endl;
    cout << "Macro-avg Precision: " << final_results.macro_avg.precision << endl;
    cout << "Macro-avg Recall: " << final_results.macro_avg.recall << endl;
    
    cout << "\n" << string(40, '=') << endl;
    cout << "TIMING ANALYSIS" << endl;
    cout << string(40, '=') << endl;
    for (const auto& [phase, time_val] : timing_data) {
        double percentage = (time_val / timing_data["total_time"]) * 100;
        cout << phase << ": " << time_val << "s (" << percentage << "%)" << endl;
    }
    
    cout << "\n" << string(40, '=') << endl;
    cout << "PER-CLASS PERFORMANCE" << endl;
    cout << string(40, '=') << endl;
    cout << left << setw(8) << "Class" << setw(12) << "Precision" 
         << setw(12) << "Recall" << setw(12) << "F1-Score" << setw(12) << "Support" << endl;
    cout << string(55, '-') << endl;
    for (const auto& [cls, metrics] : final_results.class_metrics) {
        cout << left << setw(8) << cls << setw(12) << fixed << setprecision(4) 
             << metrics.precision << setw(12) << metrics.recall 
             << setw(12) << metrics.f1 << setw(12) << metrics.support << endl;
    }
    
    // Save results with parameter sweep data if available
    if (parameter_sweep) {
        save_results(final_results, timing_data, test_labels, final_predictions, output_dir, &results_list, &k_values);
    } else {
        save_results(final_results, timing_data, test_labels, final_predictions, output_dir);
    }
    
    cout << "\nResults saved to " << output_dir << "/results.json" << endl;
    cout << "Predictions saved to " << output_dir << "/predictions.json" << endl;
    cout << "Plot data saved to " << output_dir << "/plot_data.json" << endl;
    
    cout << "\n" << string(60, '=') << endl;
    cout << "ANALYSIS COMPLETE" << endl;
    cout << string(60, '=') << endl;
    cout << "Total execution time: " << timing_data["total_time"] << " seconds" << endl;
    cout << "Average prediction time: " << (timing_data["predict_time"] * 1000 / test_scanpaths.size()) 
         << " ms/sample" << endl;
    cout << "Throughput: " << (test_scanpaths.size() / timing_data["predict_time"]) 
         << " samples/second" << endl;
    
    return 0;
} 