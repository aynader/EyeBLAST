#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>
#include "fast_eyeblast.h"

using json = nlohmann::json;
using namespace pp_eyeblast;

// Test fixture for equivalence testing
class EquivalenceTest {
private:
    FastEyeBLAST eyeblast;
    
public:
    EquivalenceTest() : eyeblast(3, 5, true, 8, 64) {}
    
    bool test_token_encoding() {
        std::cout << "Testing token encoding..." << std::endl;
        
        std::vector<std::string> tokens = {"F250", "S100", "F300", "S50", "F200"};
        
        // Test encoding
        Encoding result = eyeblast.encode_and_pack(tokens);
        
        // Basic sanity checks
        assert(result.ids.size() == tokens.size());
        assert(!result.packed.empty());
        assert(result.simhash != 0);
        
        std::cout << "  - Encoded " << tokens.size() << " tokens" << std::endl;
        std::cout << "  - Generated " << result.packed.size() << " packed words" << std::endl;
        std::cout << "  - SimHash: 0x" << std::hex << result.simhash << std::dec << std::endl;
        
        return true;
    }
    
    bool test_similarity_calculation() {
        std::cout << "Testing similarity calculation..." << std::endl;
        
        // Test identical sequences
        std::vector<int> seq1 = {0, 1, 2, 3, 4};
        std::vector<int> seq2 = {0, 1, 2, 3, 4};
        
        double sim1 = eyeblast.normalized_similarity(seq1, seq2, 5);
        std::cout << "  - Identical sequences similarity: " << sim1 << std::endl;
        
        // Test different sequences
        std::vector<int> seq3 = {0, 1, 2};
        std::vector<int> seq4 = {5, 6, 7};
        
        double sim2 = eyeblast.normalized_similarity(seq3, seq4, 5);
        std::cout << "  - Different sequences similarity: " << sim2 << std::endl;
        
        // Basic sanity checks
        assert(sim1 >= 0.0 && sim1 <= 1.0);
        assert(sim2 >= 0.0 && sim2 <= 1.0);
        assert(sim1 > sim2);  // Identical should be more similar
        
        return true;
    }
    
    bool test_hamming_distance() {
        std::cout << "Testing Hamming distance..." << std::endl;
        
        uint64_t a = 0b1010101010101010;
        uint64_t b = 0b1100110011001100;
        
        int dist = eyeblast.hamming_distance(a, b);
        std::cout << "  - Hamming distance: " << dist << std::endl;
        
        assert(dist >= 0);
        assert(eyeblast.hamming_distance(a, a) == 0);
        
        return true;
    }
    
    bool test_with_json_data() {
        std::cout << "Testing with JSON data..." << std::endl;
        
        // Try to load a sample JSON file
        std::string json_file = "PythonVsCpp/GazeBaseVR_IVT/S_1002_S1_1_VRG_tokens.json";
        std::ifstream file(json_file);
        
        if (!file.is_open()) {
            std::cout << "  - Warning: Could not open " << json_file << std::endl;
            return true;  // Skip this test if file not found
        }
        
        json j;
        file >> j;
        
        std::vector<std::string> tokens = j.get<std::vector<std::string>>();
        
        if (tokens.empty()) {
            std::cout << "  - Warning: Empty token list" << std::endl;
            return true;
        }
        
        // Test encoding
        Encoding result = eyeblast.encode_and_pack(tokens);
        
        std::cout << "  - Loaded " << tokens.size() << " tokens from JSON" << std::endl;
        std::cout << "  - Generated " << result.packed.size() << " packed words" << std::endl;
        std::cout << "  - SimHash: 0x" << std::hex << result.simhash << std::dec << std::endl;
        
        // Test similarity with itself
        double self_sim = eyeblast.normalized_similarity(result.ids, result.ids, 5);
        std::cout << "  - Self-similarity: " << self_sim << std::endl;
        
        assert(self_sim >= 0.0 && self_sim <= 1.0);
        
        return true;
    }
    
    bool test_deterministic_behavior() {
        std::cout << "Testing deterministic behavior..." << std::endl;
        
        std::vector<std::string> tokens = {"F100", "S50", "F200", "S75", "F150"};
        
        // Run encoding multiple times
        Encoding result1 = eyeblast.encode_and_pack(tokens);
        Encoding result2 = eyeblast.encode_and_pack(tokens);
        
        // Should be identical
        assert(result1.ids == result2.ids);
        assert(result1.packed == result2.packed);
        assert(result1.simhash == result2.simhash);
        
        std::cout << "  - Deterministic behavior verified" << std::endl;
        
        return true;
    }
};

int main() {
    std::cout << "=== FastEyeBLAST Equivalence Tests ===" << std::endl;
    
    EquivalenceTest test;
    bool all_passed = true;
    
    try {
        all_passed &= test.test_token_encoding();
        all_passed &= test.test_similarity_calculation();
        all_passed &= test.test_hamming_distance();
        all_passed &= test.test_with_json_data();
        all_passed &= test.test_deterministic_behavior();
        
        std::cout << std::endl;
        if (all_passed) {
            std::cout << "✓ All tests passed!" << std::endl;
            return 0;
        } else {
            std::cout << "✗ Some tests failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}