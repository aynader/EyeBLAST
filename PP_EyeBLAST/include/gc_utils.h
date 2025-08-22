#pragma once
#include <vector>
#include <emp-tool/emp-tool.h>

// Use EMP-toolkit's built-in hamming_weight() function instead of custom implementation
inline emp::Integer PopcountCircuit(const std::vector<emp::Bit>& bits) {
    using namespace emp;
    if (bits.empty())
        return Integer(64, 0, PUBLIC);

    // Create an Integer from the bit vector and use built-in hamming_weight()
    Integer bit_integer(bits.size(), 0, PUBLIC);
    for (size_t i = 0; i < bits.size(); ++i) {
        bit_integer[i] = bits[i];
    }
    
    return bit_integer.hamming_weight();
}
