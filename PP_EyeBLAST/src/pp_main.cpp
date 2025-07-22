#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <set>
#include <map>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <nlohmann/json.hpp>
#include <emp-tool/emp-tool.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include "fast_eyeblast.h"

using namespace pp_eyeblast;
using namespace emp;
using json = nlohmann::json;
namespace fs = std::filesystem;


int run_command(const std::string& command) {
    std::cout << "Running: " << command << std::endl;
    return system(command.c_str());
}


std::string read_result_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open result file: " + filename);
    }
    
    std::string result;
    std::getline(file, result);
    file.close();
    
    return result;
}


struct ScanpathFile {
    std::string filepath;
    std::string label;
    int index;
};


std::vector<ScanpathFile> enumerate_json_files(const std::string& directory) {
    std::vector<ScanpathFile> files;
    
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory);
    }
    
    int index = 0;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            ScanpathFile file;
            file.filepath = entry.path().string();
            file.index = index++;
            
            
            std::string filename = entry.path().stem().string();
            size_t underscore_pos = filename.find('_');
            if (underscore_pos != std::string::npos) {
                file.label = filename.substr(0, underscore_pos);
            } else {
                file.label = filename;
            }
            
            files.push_back(file);
        }
    }
    
    
    std::sort(files.begin(), files.end(), 
              [](const ScanpathFile& a, const ScanpathFile& b) {
                  return a.filepath < b.filepath;
              });
    
    return files;
}


struct SimilarityResult {
    double similarity_score;
    std::string label;
    int index;
    std::string filepath;
};


void run_pp_protocol_1to1(int party, const std::string& ip, int coarse_port, int refine_port,
                         const std::string& alice_file, const std::string& bob_file, 
                         int tau, int band_width, const std::string& output_file);


std::string knn_vote(const std::vector<SimilarityResult>& similarities, int k) {
    if (similarities.empty()) {
        return "unknown";
    }
    
    
    std::vector<SimilarityResult> sorted_similarities = similarities;
    std::sort(sorted_similarities.begin(), sorted_similarities.end(),
              [](const SimilarityResult& a, const SimilarityResult& b) {
                  return a.similarity_score > b.similarity_score;
              });
    
    
    int actual_k = std::min(k, static_cast<int>(sorted_similarities.size()));
    
    
    std::map<std::string, double> label_weights;
    for (int i = 0; i < actual_k; i++) {
        const auto& neighbor = sorted_similarities[i];
        label_weights[neighbor.label] += neighbor.similarity_score;
    }
    
    
    std::string best_label = "unknown";
    double best_weight = -1.0;
    for (const auto& pair : label_weights) {
        if (pair.second > best_weight) {
            best_weight = pair.second;
            best_label = pair.first;
        }
    }
    
    return best_label;
}


void run_pp_protocol_1toN(int party, const std::string& ip, int coarse_port, int refine_port,
                         const std::string& alice_file, const std::string& bob_dir, 
                         int tau, int band_width, int k_neighbors, int max_candidates,
                         const std::string& output_file) {
    
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== Privacy-Preserving EyeBLAST 1-to-N Protocol ===" << std::endl;
    std::cout << "Party: " << (party == ALICE ? "Alice" : "Bob") << std::endl;
    
    
    if (party == ALICE) {
        std::cout << "Alice file: " << alice_file << std::endl;
        std::cout << "Bob directory: <hidden for privacy>" << std::endl;
    } else {
        std::cout << "Alice file: <hidden for privacy>" << std::endl;
        std::cout << "Bob directory: " << bob_dir << std::endl;
    }
    
    std::cout << "k-neighbors: " << k_neighbors << std::endl;
    std::cout << "Max candidates for refinement: " << max_candidates << std::endl;
    std::cout << "Coarse threshold: " << tau << std::endl;
    std::cout << "Band width: " << band_width << std::endl;
    
    
    std::vector<ScanpathFile> bob_files;
    
    if (party == ALICE) {
        
        std::cout << "Alice query: " << alice_file << std::endl;
        std::cout << "Bob dataset: <size hidden for privacy>" << std::endl;
    } else {
        
        bob_files = enumerate_json_files(bob_dir);
        if (bob_files.empty()) {
            throw std::runtime_error("No JSON files found in Bob directory: " + bob_dir);
        }
        std::cout << "Alice query: <hidden for privacy>" << std::endl;
        std::cout << "Bob dataset: " << bob_files.size() << " files" << std::endl;
        
        
        if (bob_files.size() == 1) {
            std::cout << "Single file in Bob directory, using 1-to-1 protocol..." << std::endl;
            
            return run_pp_protocol_1to1(party, ip, coarse_port, refine_port,
                                       "", bob_files[0].filepath, tau, band_width, output_file);
        }
    }
    
    
    std::vector<SimilarityResult> similarities;
    std::string predicted_label = "unknown";
    
    if (party == BOB) {
        std::cout << "\\n=== Bob participating in Alice-orchestrated 1-to-N protocol ===" << std::endl;
        
        
        for (size_t i = 0; i < bob_files.size(); i++) {
            std::cout << "Waiting for Alice on port " << (coarse_port + static_cast<int>(i)) 
                      << " for file " << bob_files[i].filepath << std::endl;
            
            try {
                std::string coarse_cmd = "./pp_coarse --role bob --ip " + ip + 
                                       " --port " + std::to_string(coarse_port + static_cast<int>(i)) +
                                       " --infile " + bob_files[i].filepath +
                                       " --tau " + std::to_string(tau) +
                                       " --output temp_coarse_bob_" + std::to_string(i) + ".txt";
                
                int coarse_result = run_command(coarse_cmd);
                if (coarse_result != 0) {
                    std::cout << " -> ERROR" << std::endl;
                    continue;
                }
                
                std::cout << " -> COMPLETED" << std::endl;
                std::remove(("temp_coarse_bob_" + std::to_string(i) + ".txt").c_str());
                
            } catch (const std::exception& e) {
                std::cerr << " -> ERROR: " << e.what() << std::endl;
                continue;
            }
        }
        
        
        std::cout << "\\n=== Bob participating in fine refinement ===" << std::endl;
        int refinement_round = 0;
        while (refinement_round < max_candidates) {
            std::cout << "Waiting for Alice on refinement port " << (refine_port + refinement_round) << std::endl;
            
            
            int file_index = refinement_round % bob_files.size();
            std::string refine_cmd = "./pp_refine --role bob --ip " + ip + 
                                   " --port " + std::to_string(refine_port + refinement_round) +
                                   " --infile " + bob_files[file_index].filepath +
                                   " --band_w " + std::to_string(band_width) +
                                   " --output temp_refine_bob_" + std::to_string(refinement_round) + ".txt";
            
            int refine_result = run_command(refine_cmd);
            if (refine_result != 0) {
                std::cout << "Alice finished refinement after " << refinement_round << " rounds" << std::endl;
                break; 
            }
            
            std::cout << "Refinement round " << (refinement_round + 1) << " completed" << std::endl;
            std::remove(("temp_refine_bob_" + std::to_string(refinement_round) + ".txt").c_str());
            refinement_round++;
        }
        
        std::cout << "Bob completed protocol participation" << std::endl;
        
    } else {
        
        std::cout << "\\n=== Alice orchestrating 1-to-N protocol ===" << std::endl;
        
        
        std::cout << "Phase 1: Coarse filtering..." << std::endl;
        std::vector<int> passed_indices;
        
        for (int i = 0; i < 1000; i++) {  
            std::cout << "Connecting to Bob on port " << (coarse_port + i) << " for sample " << (i+1) << "..." << std::endl;
            
            std::string coarse_cmd = "./pp_coarse --role alice --ip " + ip + 
                                   " --port " + std::to_string(coarse_port + i) +
                                   " --infile " + alice_file +
                                   " --tau " + std::to_string(tau) +
                                   " --output temp_coarse_alice_" + std::to_string(i) + ".txt";
            
            
            std::string timeout_cmd = "timeout 10 " + coarse_cmd;
            int coarse_result = run_command(timeout_cmd);
            if (coarse_result != 0) {
                
                std::cout << "Bob finished coarse filtering after " << i << " samples" << std::endl;
                break;
            }
            
            
            std::string coarse_output = read_result_file("temp_coarse_alice_" + std::to_string(i) + ".txt");
            bool passed = (coarse_output == "1");
            
            if (passed) {
                passed_indices.push_back(i);
                std::cout << "Sample " << (i+1) << " -> PASS" << std::endl;
            } else {
                std::cout << "Sample " << (i+1) << " -> FAIL" << std::endl;
            }
            
            std::remove(("temp_coarse_alice_" + std::to_string(i) + ".txt").c_str());
        }
        
        std::cout << "Coarse filtering complete: " << passed_indices.size() << " samples passed" << std::endl;
        
        if (passed_indices.empty()) {
            predicted_label = "no_match";
        } else {
            
            std::cout << "Phase 2: Fine refinement..." << std::endl;
            int refinement_count = std::min(static_cast<int>(passed_indices.size()), max_candidates);
            
            for (int i = 0; i < refinement_count; i++) {
                std::cout << "Connecting to Bob on refinement port " << (refine_port + i) 
                          << " for passed sample " << (passed_indices[i] + 1) << "..." << std::endl;
                
                std::string refine_cmd = "./pp_refine --role alice --ip " + ip + 
                                       " --port " + std::to_string(refine_port + i) +
                                       " --infile " + alice_file +
                                       " --band_w " + std::to_string(band_width) +
                                       " --output temp_refine_alice_" + std::to_string(i) + ".txt";
                
                int refine_result = run_command(refine_cmd);
                if (refine_result != 0) {
                    std::cerr << "Fine refinement failed for sample " << (passed_indices[i] + 1) << std::endl;
                    continue;
                }
                
                
                std::string refine_output = read_result_file("temp_refine_alice_" + std::to_string(i) + ".txt");
                std::istringstream iss(refine_output);
                int raw_score;
                size_t alice_len, bob_len;
                
                if (iss >> raw_score >> alice_len >> bob_len) {
                    int max_possible_score = std::max(alice_len, bob_len) * 2;
                    double normalized_score = std::max(0.0, static_cast<double>(raw_score)) / max_possible_score;
                    
                    SimilarityResult sim_result;
                    sim_result.similarity_score = normalized_score;
                    sim_result.label = "sample_" + std::to_string(passed_indices[i]);
                    sim_result.index = passed_indices[i];
                    sim_result.filepath = "hidden";
                    
                    similarities.push_back(sim_result);
                    
                    std::cout << "Sample " << (passed_indices[i]+1) << " -> Similarity: " << normalized_score << std::endl;
                }
                
                std::remove(("temp_refine_alice_" + std::to_string(i) + ".txt").c_str());
            }
            
            
            if (!similarities.empty()) {
                predicted_label = knn_vote(similarities, k_neighbors);
            }
        }
        
        std::cout << "Alice completed 1-to-N orchestration" << std::endl;
    }
    
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (party == ALICE) {
        std::cout << "\\n=== k-NN Classification Result ===" << std::endl;
        std::cout << "Predicted label: " << predicted_label << std::endl;
        std::cout << "Total protocol time: " << duration.count() << " ms" << std::endl;
        
        
        json alice_result;
        alice_result["predicted_label"] = predicted_label;
        alice_result["total_time_ms"] = duration.count();
        alice_result["privacy_preserving"] = true;
        alice_result["protocol"] = "1-to-N";
        
        
        json sim_array = json::array();
        for (const auto& sim : similarities) {
            json sim_obj;
            sim_obj["similarity_score"] = sim.similarity_score;
            sim_obj["label"] = sim.label;
            sim_obj["index"] = sim.index;
            sim_array.push_back(sim_obj);
        }
        alice_result["similarities"] = sim_array;
        
        std::ofstream alice_out(output_file);
        alice_out << alice_result.dump(2) << std::endl;
        alice_out.close();
        
        std::cout << "Results saved to: " << output_file << std::endl;
        
    } else {
        std::cout << "\\n=== Protocol Completed ===" << std::endl;
        std::cout << "Total protocol time: " << duration.count() << " ms" << std::endl;
        std::cout << "Bob learns nothing (privacy-preserving)" << std::endl;
    }
    
    std::cout << "\\n=== 1-to-N Protocol Completed Successfully ===" << std::endl;
}


void run_pp_protocol_1to1(int party, const std::string& ip, int coarse_port, int refine_port,
                         const std::string& alice_file, const std::string& bob_file, 
                         int tau, int band_width, const std::string& output_file) {
    
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== Privacy-Preserving EyeBLAST 1-to-1 Protocol ===" << std::endl;
    std::cout << "Party: " << (party == ALICE ? "Alice" : "Bob") << std::endl;
    std::cout << "Alice file: " << alice_file << std::endl;
    std::cout << "Bob file: " << bob_file << std::endl;
    std::cout << "Coarse threshold: " << tau << std::endl;
    std::cout << "Band width: " << band_width << std::endl;
    
    
    std::cout << "\\nLoading scanpath..." << std::endl;
    std::string infile = (party == ALICE) ? alice_file : bob_file;
    std::ifstream file(infile);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + infile);
    }
    
    json j;
    file >> j;
    std::vector<std::string> tokens = j.get<std::vector<std::string>>();
    file.close();
    
    if (tokens.empty()) {
        throw std::runtime_error("Empty token list");
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
    
    
    std::cout << "Encoding scanpath..." << std::endl;
    Encoding encoding = encode_and_pack(tokens);
    
    std::cout << "Generated SimHash: 0x" << std::hex << encoding.simhash << std::dec << std::endl;
    std::cout << "Generated " << encoding.packed.size() << " packed words" << std::endl;
    
    
    std::cout << "\\n=== Phase 1: Coarse Filter ===" << std::endl;
    
    std::string coarse_result_file = "coarse_result_" + std::to_string(party) + ".txt";
    std::string coarse_command = "./pp_coarse --role " + 
                                (party == ALICE ? std::string("alice") : std::string("bob")) +
                                " --ip " + ip +
                                " --port " + std::to_string(coarse_port) +
                                " --infile " + infile +
                                " --tau " + std::to_string(tau) +
                                " --output " + coarse_result_file;
    
    int coarse_result = run_command(coarse_command);
    if (coarse_result != 0) {
        throw std::runtime_error("Coarse filter failed with exit code: " + std::to_string(coarse_result));
    }
    
    
    bool pass_coarse = false;
    if (party == ALICE) {
        std::string coarse_output = read_result_file(coarse_result_file);
        pass_coarse = (coarse_output == "1");
        
        std::cout << "Coarse filter result: " << (pass_coarse ? "PASS" : "FAIL") << std::endl;
        
        if (!pass_coarse) {
            std::cout << "Coarse filter failed. Final similarity score: 0.0" << std::endl;
            
            
            json final_result;
            final_result["score"] = 0.0;
            final_result["coarse_passed"] = false;
            final_result["raw_score"] = 0;
            
            std::ofstream out(output_file);
            out << final_result.dump(2) << std::endl;
            out.close();
            
            return;
        }
    }
    
    
    std::cout << "\\n=== Phase 2: Fine Refinement ===" << std::endl;
    
    std::string refine_result_file = "refine_result_" + std::to_string(party) + ".txt";
    std::string refine_command = "./pp_refine --role " + 
                                (party == ALICE ? std::string("alice") : std::string("bob")) +
                                " --ip " + ip +
                                " --port " + std::to_string(refine_port) +
                                " --infile " + infile +
                                " --band_w " + std::to_string(band_width) +
                                " --output " + refine_result_file;
    
    int refine_result = run_command(refine_command);
    if (refine_result != 0) {
        throw std::runtime_error("Fine refinement failed with exit code: " + std::to_string(refine_result));
    }
    
    
    if (party == ALICE) {
        std::string refine_output = read_result_file(refine_result_file);
        std::istringstream iss(refine_output);
        int raw_score;
        size_t alice_len, bob_len;
        
        if (!(iss >> raw_score >> alice_len >> bob_len)) {
            throw std::runtime_error("Invalid refinement result format");
        }
        
        std::cout << "Raw refinement score: " << raw_score << std::endl;
        std::cout << "Alice sequence length (tokens): " << alice_len << std::endl;
        std::cout << "Bob sequence length (tokens): " << bob_len << std::endl;
        
        
        
        int max_possible_score = std::max(alice_len, bob_len) * 2;
        double normalized_score = std::max(0.0, static_cast<double>(raw_score)) / max_possible_score;
        
        std::cout << "Max possible score: " << max_possible_score << std::endl;
        std::cout << "Normalized similarity score: " << normalized_score << std::endl;
        
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Total protocol time: " << duration.count() << " ms" << std::endl;
        
        
        json final_result;
        final_result["score"] = normalized_score;
        final_result["coarse_passed"] = true;
        final_result["raw_score"] = raw_score;
        final_result["max_possible_score"] = max_possible_score;
        final_result["processed_words"] = std::max(alice_len, bob_len);  
        final_result["alice_packed_words"] = alice_len;
        final_result["bob_packed_words"] = bob_len;
        final_result["total_time_ms"] = duration.count();
        final_result["tokens_processed"] = tokens.size();
        final_result["simhash"] = encoding.simhash;
        final_result["packed_words"] = encoding.packed.size();
        
        std::ofstream out(output_file);
        out << final_result.dump(2) << std::endl;
        out.close();
        
        std::cout << "Results saved to: " << output_file << std::endl;
    }
    
    std::cout << "\\n=== Protocol Completed Successfully ===" << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --role {alice|bob}     Party role (required)" << std::endl;
    std::cout << "  --infile PATH          Input file/directory path (required)" << std::endl;
    std::cout << "  --ip IP_ADDRESS        IP address for Bob to connect to Alice (default: 127.0.0.1)" << std::endl;
    std::cout << "  --coarse_port PORT     Port for coarse filter (default: 12345)" << std::endl;
    std::cout << "  --refine_port PORT     Port for refinement (default: 12346)" << std::endl;
    std::cout << "  --k_neighbors K        Number of neighbors for k-NN voting (default: 5)" << std::endl;
    std::cout << "  --max_candidates N     Maximum candidates for fine refinement (default: 20)" << std::endl;
    std::cout << "  --tau THRESHOLD        Hamming distance threshold (default: 20)" << std::endl;
    std::cout << "  --band_w WIDTH         Band width for DP (default: 5)" << std::endl;
    std::cout << "  --output FILE          Output file for result (default: pp_result.json)" << std::endl;
    std::cout << "  --help                 Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example usage:" << std::endl;
    std::cout << "  Alice: " << program_name << " --role alice --infile ../Alice --output alice_out.json" << std::endl;
    std::cout << "  Bob:   " << program_name << " --role bob --infile ../Bob --ip 127.0.0.1" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: Protocol automatically detects 1-to-1 vs 1-to-N based on Bob's dataset size" << std::endl;
}

int main(int argc, char* argv[]) {
    
    int party = -1;
    std::string ip = "127.0.0.1";
    int coarse_port = 12345;
    int refine_port = 12346;
    std::string infile;
    int k_neighbors = 5;
    int max_candidates = 20;
    int tau = 20;  
    int band_width = 5;
    std::string output_file = "pp_result.json";
    
    
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
        } else if (arg == "--infile" && i + 1 < argc) {
            infile = argv[++i];
        } else if (arg == "--ip" && i + 1 < argc) {
            ip = argv[++i];
        } else if (arg == "--coarse_port" && i + 1 < argc) {
            coarse_port = std::stoi(argv[++i]);
        } else if (arg == "--refine_port" && i + 1 < argc) {
            refine_port = std::stoi(argv[++i]);
        } else if (arg == "--k_neighbors" && i + 1 < argc) {
            k_neighbors = std::stoi(argv[++i]);
        } else if (arg == "--max_candidates" && i + 1 < argc) {
            max_candidates = std::stoi(argv[++i]);
        } else if (arg == "--tau" && i + 1 < argc) {
            tau = std::stoi(argv[++i]);
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
        
        std::cout << "=== Phase 0: Protocol Coordination ===" << std::endl;
        
        
        std::string coord_output = "coordinate_result.txt";
        int coord_port = coarse_port - 1; 
        
        std::string coord_cmd = "./pp_coordinate --role " + std::string(party == ALICE ? "alice" : "bob") +
                               " --ip " + ip + " --port " + std::to_string(coord_port) +
                               (party == BOB ? " --bob_dir " + infile : "") +
                               " --output " + coord_output;
        
        int coord_result = run_command(coord_cmd);
        if (coord_result != 0) {
            throw std::runtime_error("Protocol coordination failed");
        }
        
        
        std::string party_coord_output = coord_output + "_" + (party == ALICE ? "alice" : "bob");
        std::ifstream coord_file(party_coord_output);
        if (!coord_file.is_open()) {
            throw std::runtime_error("Could not read coordination result");
        }
        
        std::string protocol_choice;
        std::getline(coord_file, protocol_choice);
        coord_file.close();
        
        
        std::remove(party_coord_output.c_str());
        
        std::cout << "Coordination complete: Using " << protocol_choice << " protocol" << std::endl;
        
        
        std::string alice_file;
        if (party == ALICE) {
            if (fs::is_directory(infile)) {
                std::vector<ScanpathFile> alice_files = enumerate_json_files(infile);
                if (alice_files.empty()) {
                    throw std::runtime_error("No JSON files found in Alice directory: " + infile);
                }
                if (alice_files.size() > 1) {
                    std::cout << "Warning: Multiple files in Alice directory, using first one: " 
                              << alice_files[0].filepath << std::endl;
                }
                alice_file = alice_files[0].filepath;
            } else {
                alice_file = infile;
            }
        }
        
        
        if (protocol_choice == "1to1") {
            std::cout << "=== Running 1-to-1 Protocol ===" << std::endl;
            if (party == ALICE) {
                run_pp_protocol_1to1(party, ip, coarse_port, refine_port, alice_file, "", tau, band_width, output_file);
            } else {
                
                if (fs::is_directory(infile)) {
                    std::vector<ScanpathFile> bob_files = enumerate_json_files(infile);
                    run_pp_protocol_1to1(party, ip, coarse_port, refine_port, "", bob_files[0].filepath, tau, band_width, output_file);
                } else {
                    run_pp_protocol_1to1(party, ip, coarse_port, refine_port, "", infile, tau, band_width, output_file);
                }
            }
        } else {
            std::cout << "=== Running 1-to-N Protocol ===" << std::endl;
            if (party == ALICE) {
                run_pp_protocol_1toN(party, ip, coarse_port, refine_port, alice_file, "", tau, band_width, k_neighbors, max_candidates, output_file);
            } else {
                run_pp_protocol_1toN(party, ip, coarse_port, refine_port, "", infile, tau, band_width, k_neighbors, max_candidates, output_file);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}