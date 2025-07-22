#pragma once
#include <vector>
#include <emp-tool/emp-tool.h>


inline emp::Integer PopcountCircuit(const std::vector<emp::Bit>& bits) {
    using namespace emp;
    if (bits.empty())
        return Integer(64, 0, PUBLIC);

    
    Integer bit_integer(bits.size(), 0, PUBLIC);
    for (size_t i = 0; i < bits.size(); ++i) {
        bit_integer[i] = bits[i];
    }
    
    return bit_integer.hamming_weight();
}
