#include <iostream>
#include <fstream>
#include <filesystem>
#include <emp-tool/emp-tool.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include "fast_eyeblast.h"

using namespace emp;
using namespace pp_eyeblast;
namespace fs = std::filesystem;


void run_coordinate_gc(int party, const std::string& ip, int port, 
                      const std::string& bob_dir, const std::string& output_file) {
    
    std::cout << "=== Protocol Coordination Phase ===" << std::endl;
    std::cout << "Party: " << (party == ALICE ? "Alice" : "Bob") << std::endl;
    
    
    NetIO io(party == ALICE ? nullptr : ip.c_str(), port);
    setup_semi_honest(&io, party);
    
    std::cout << "Connected. Starting coordination circuit..." << std::endl;
    
    
    int bob_file_count = 0;
    if (party == BOB) {
        if (fs::is_directory(bob_dir)) {
            for (const auto& entry : fs::directory_iterator(bob_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    bob_file_count++;
                }
            }
        } else {
            bob_file_count = 1; 
        }
        std::cout << "Bob has " << bob_file_count << " files" << std::endl;
    }
    
    
    Integer bob_count(32, bob_file_count, BOB);
    Integer threshold(32, 1, PUBLIC);
    
    
    Bit use_1to1 = bob_count.equal(threshold);
    
    
    bool protocol_1to1 = use_1to1.reveal<bool>();
    
    std::cout << "Coordination result: " << (protocol_1to1 ? "1-to-1" : "1-to-N") << " protocol" << std::endl;
    
    finalize_semi_honest();
    
    
    std::string party_output = output_file + "_" + (party == ALICE ? "alice" : "bob");
    std::ofstream out(party_output);
    out << (protocol_1to1 ? "1to1" : "1toN") << std::endl;
    out.close();
    
    std::cout << "Coordination completed!" << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --role {alice|bob}     Party role (required)" << std::endl;
    std::cout << "  --ip IP_ADDRESS        IP address for Bob to connect to Alice" << std::endl;
    std::cout << "  --port PORT            Port number (default: 12340)" << std::endl;
    std::cout << "  --bob_dir DIR          Bob's directory (Bob only)" << std::endl;
    std::cout << "  --output FILE          Output file for result (default: coordinate_result.txt)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    
    int party = -1;
    std::string ip = "127.0.0.1";
    int port = 12340;
    std::string bob_dir = "";
    std::string output_file = "coordinate_result.txt";
    
    
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
        } else if (arg == "--bob_dir" && i + 1 < argc) {
            bob_dir = argv[++i];
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
    
    if (party == BOB && bob_dir.empty()) {
        std::cerr << "Error: --bob_dir is required for Bob" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        run_coordinate_gc(party, ip, port, bob_dir, output_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}