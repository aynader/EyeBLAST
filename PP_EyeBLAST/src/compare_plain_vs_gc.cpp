// src/compare_plain.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

#include "fast_eyeblast.h"      // the library you already built

using json = nlohmann::json;
using namespace pp_eyeblast;

static std::vector<std::string> read_tokens(const std::string &fname) {
    std::ifstream f(fname);
    if (!f.is_open()) throw std::runtime_error("cannot open " + fname);
    json j;  f >> j;
    return j.get<std::vector<std::string>>();
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " alice_tokens.json bob_tokens.json [band_width=5]\n";
        return 1;
    }
    int band_w = (argc == 4) ? std::stoi(argv[3]) : 5;

    auto alice_tok = read_tokens(argv[1]);
    auto bob_tok   = read_tokens(argv[2]);

    // use the global FastEyeBLAST instance
    Encoding encA = encode_and_pack(alice_tok);
    Encoding encB = encode_and_pack(bob_tok);

    std::cout << "DEBUG: Alice tokens=" << alice_tok.size() << ", ids=" << encA.ids.size() << ", packed=" << encA.packed.size() << std::endl;
    std::cout << "DEBUG: Bob tokens=" << bob_tok.size() << ", ids=" << encB.ids.size() << ", packed=" << encB.packed.size() << std::endl;
    
    // Check if sequences are identical
    bool identical = (encA.ids == encB.ids);
    std::cout << "DEBUG: Sequences identical: " << (identical ? "YES" : "NO") << std::endl;

    double score = normalized_similarity(encA.ids, encB.ids, band_w);
    std::cout << "Plain EyeBLAST similarity: " << score << "\n";
    return 0;
}
