import json
import time
from Fast_EyeBLAST_IVT import FastEyeBLAST  # Adjust import if needed

# Load the two scanpaths
with open('../Alice/query.json') as f:
    scanpath1 = json.load(f)
with open('../Bob_Single/S_1002_S1_1_VRG_tokens.json') as f:
    scanpath2 = json.load(f)

# Initialize FastEyeBLAST
eyeblast = FastEyeBLAST()

# Start timing
start_time = time.perf_counter()

# Register tokens by encoding both scanpaths
encoded1 = eyeblast._encode_scanpath(scanpath1)
encoded2 = eyeblast._encode_scanpath(scanpath2)

# Initialize bit-packing (required for similarity)
eyeblast._initialize_bit_packing()

# Compute similarity score
score = eyeblast._similarity_score(encoded1, encoded2)

# Stop timing
end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"Similarity score between the two scanpaths: {score}")
print(f"Time taken: {elapsed:.4f} seconds")
