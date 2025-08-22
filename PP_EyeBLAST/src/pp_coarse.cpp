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

// PopcountCircuit is now defined in gc_utils.h

void run_coarse_gc(int party, const std::string& ip, int port, 
                   uint64_t local_simhash, int tau, const std::string& output_file) {
    
    std::cout << "Setting up EMP-Toolkit for party " << party << std::endl;
    
    // Setup network I/O
    NetIO io(party == ALICE ? nullptr : ip.c_str(), port);
    // EMP-Toolkit setup for semi-honest security
    setup_semi_honest(&io, party);
    
    std::cout << "Connected. Starting coarse filter circuit..." << std::endl;
    
    // Prepare 64-bit inputs for Alice and Bob
    std::vector<Bit> alice_bits(64);
    std::vector<Bit> bob_bits(64);
    
    for (int i = 0; i < 64; i++) {
        bool alice_bit = (party == ALICE) ? ((local_simhash >> i) & 1) : false;
        bool bob_bit = (party == BOB) ? ((local_simhash >> i) & 1) : false;
        
        alice_bits[i] = Bit(alice_bit, ALICE);
        bob_bits[i] = Bit(bob_bit, BOB);
    }
    
    std::cout << "Computing XOR differences..." << std::endl;
    
    // XOR each bit position
    std::vector<Bit> diff_bits(64);
    for (int i = 0; i < 64; i++) {
        diff_bits[i] = alice_bits[i] ^ bob_bits[i];
    }
    
    // DEBUG: Both parties reveal XOR bits
    std::cout << "=== DEBUG: XOR BITS (Party " << (party == ALICE ? "Alice" : "Bob") << ") ===" << std::endl;
    std::cout << "XOR bits: ";
    for (int i = 0; i < 64; i++) {
        // Both parties reveal the XOR result
        bool bit_val = diff_bits[i].reveal<bool>();
        std::cout << (bit_val ? "1" : "0");
        if (i % 8 == 7) std::cout << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Computing popcount via circuit..." << std::endl;
    
    // Count set bits using popcount circuit
    Integer hamming_distance = PopcountCircuit(diff_bits);
    
    // DEBUG: Both parties reveal hamming distance
    std::cout << "=== DEBUG: HAMMING DISTANCE (Party " << (party == ALICE ? "Alice" : "Bob") << ") ===" << std::endl;
    int hamming_val = hamming_distance.reveal<int>();
    std::cout << "Hamming distance: " << hamming_val << std::endl;
    
    std::cout << "Comparing against threshold..." << std::endl;
    
    // Compare against threshold (hamming_distance <= threshold) - EXACT same as original
    Integer threshold(64, tau, PUBLIC);
    Bit pass_coarse = threshold.geq(hamming_distance);
    
    std::cout << "Revealing result..." << std::endl;
    
    // Reveal result to Alice
    bool result = pass_coarse.reveal<bool>(ALICE);
    
    std::cout << "Finalizing protocol..." << std::endl;
    finalize_semi_honest();
    
    // Save result
    if (party == ALICE) {
        std::ofstream out(output_file);
        out << (result ? "1" : "0") << std::endl;
        out.close();
        
        std::cout << "Coarse filter result: " << (result ? "PASS" : "FAIL") << std::endl;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --role {alice|bob}     Party role (required)" << std::endl;
    std::cout << "  --ip IP_ADDRESS        IP address for Bob to connect to Alice" << std::endl;
    std::cout << "  --port PORT            Port number (default: 12345)" << std::endl;
    std::cout << "  --infile FILE          JSON file with tokens (required)" << std::endl;
    std::cout << "  --tau THRESHOLD        Hamming distance threshold (default: 10)" << std::endl;
    std::cout << "  --output FILE          Output file for result (default: coarse_result.txt)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int party = -1;
    std::string ip = "127.0.0.1";
    int port = 12345;
    std::string infile;
    int tau = 10;
    std::string output_file = "coarse_result.txt";
    
    // Parse command line arguments
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
        } else if (arg == "--tau" && i + 1 < argc) {
            tau = std::stoi(argv[++i]);
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
    
    // Validate required arguments
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
        std::cout << "=== Privacy-Preserving EyeBLAST Coarse Filter ===" << std::endl;
        std::cout << "Party: " << (party == ALICE ? "Alice" : "Bob") << std::endl;
        std::cout << "Input file: " << infile << std::endl;
        std::cout << "Threshold: " << tau << std::endl;
        std::cout << "Port: " << port << std::endl;
        
        // Load and encode scanpath
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
        
        // CRITICAL: Synchronize token dictionary (same logic as pp_main)
        std::cout << "Synchronizing token dictionaries..." << std::endl;
        
        // Pre-populate with all possible token types deterministically
        std::set<std::string> comprehensive_vocab;
        
        // Add all possible binned duration tokens (M, F, S with bins 0-7)
        const std::vector<char> prefixes = {'M', 'F', 'S'};
        for (char prefix : prefixes) {
            for (int bin = 0; bin < 8; ++bin) {
                comprehensive_vocab.insert(std::string(1, prefix) + std::to_string(bin));
            }
        }
        
        // Also add actual tokens from current sequences
        for (const std::string& token : tokens) {
            comprehensive_vocab.insert(get_global_instance().bin_duration(token));
        }
        
        // Pre-populate token dictionary in deterministic order
        for (const std::string& token : comprehensive_vocab) {
            get_global_instance().token_to_integer(token);
        }
        
        // Freeze dictionary to prevent further changes
        get_global_instance().freeze_dictionary();
        
        std::cout << "Token dictionary synchronized and frozen with " << comprehensive_vocab.size() << " tokens" << std::endl;
        
        // Encode and pack scanpath
        std::cout << "Encoding scanpath..." << std::endl;
        Encoding encoding = encode_and_pack(tokens);
        
        std::cout << "Generated SimHash: 0x" << std::hex << encoding.simhash << std::dec << std::endl;
        
        // DEBUG: Show binary representation of SimHash
        std::cout << "=== DEBUG: LOCAL SIMHASH BITS ===" << std::endl;
        std::cout << "SimHash binary: ";
        for (int i = 0; i < 64; i++) {
            std::cout << ((encoding.simhash >> i) & 1);
            if (i % 8 == 7) std::cout << " ";
        }
        std::cout << std::endl;
        
        // Run coarse filter garbled circuit
        std::cout << "Starting garbled circuit protocol..." << std::endl;
        run_coarse_gc(party, ip, port, encoding.simhash, tau, output_file);
        
        std::cout << "Protocol completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}