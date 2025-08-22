# dashboard.py

import streamlit as st
import os
import glob
import json
import time
from Fast_EyeBLAST_IVT import FastEyeBLAST

# ——— DEFAULT CONFIG ———
DEFAULT_INPUT_DIR = "GazeBaseVR_IVT"
SAMPLE_PCT = 100
# ————————————————

st.sidebar.title("Configuration")
data_source = st.sidebar.radio("Data source", ["Directory", "Upload files"])

# --- Directory mode ---
@st.cache_data
def load_model_and_data(input_dir: str):
    """Load from a directory of JSONs, train FastEyeBLAST, and pre‐encode scanpaths."""
    if not os.path.isdir(input_dir):
        return [], [], [], None

    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if SAMPLE_PCT < 100 and paths:
        import random
        random.seed(42)
        k = int(len(paths) * SAMPLE_PCT / 100)
        paths = sorted(random.sample(paths, k))

    file_names = [os.path.basename(p) for p in paths]
    scanpaths = [json.load(open(p)) for p in paths]
    if not scanpaths:
        return [], [], [], None

    labels = [0] * len(scanpaths)
    fe = FastEyeBLAST(use_parallel=False)
    fe.fit(scanpaths, labels)
    encoded = [fe._encode_scanpath(sp) for sp in scanpaths]
    return file_names, scanpaths, encoded, fe

if data_source == "Directory":
    input_dir = st.sidebar.text_input("Scanpath directory", DEFAULT_INPUT_DIR)
    if st.sidebar.button("📂 Load directory"):
        # Clear cache so we re‑load even if the path is the same text
        load_model_and_data.clear()
        st.sidebar.success(f"Reloaded from {input_dir}")
    file_names, scanpaths, encoded_scanpaths, fe = load_model_and_data(input_dir)
    if fe is None:
        st.sidebar.error(f"No JSON files found in:\n{input_dir}")
        st.stop()

# --- Upload mode ---
else:
    uploaded = st.sidebar.file_uploader(
        "Upload JSON scanpath files",
        type="json",
        accept_multiple_files=True
    )
    if not uploaded:
        st.sidebar.warning("Upload one or more .json files")
        st.stop()
    # extract names & content
    file_names = [f.name for f in uploaded]
    scanpaths = [json.load(f) for f in uploaded]
    fe = FastEyeBLAST(use_parallel=False)
    fe.fit(scanpaths, [0] * len(scanpaths))
    encoded_scanpaths = [fe._encode_scanpath(sp) for sp in scanpaths]

# Now we have: file_names, scanpaths, encoded_scanpaths, and fe
N = len(file_names)
st.title("FastEyeBLAST On‑Demand Similarity")

# Selection UI
st.sidebar.header("Pick two scanpaths")
name_i = st.sidebar.selectbox("Scanpath A", file_names)
choices_j = [fn for fn in file_names if fn != name_i]
name_j = st.sidebar.selectbox("Scanpath B", choices_j)

i = file_names.index(name_i)
j = file_names.index(name_j)

def compute_details(i, j):
    """Compute normalized similarity, raw DP score, elapsed ms, seeds, and alignment."""
    t0 = time.perf_counter()
    s1, s2 = encoded_scanpaths[i], encoded_scanpaths[j]
    p1 = fe._pack_ids_into_words(s1)
    p2 = fe._pack_ids_into_words(s2)
    seeds = fe._extract_seeds(p1, p2)
    chain = fe._greedy_chain_seeds(seeds, len(s1), len(s2))
    dp_score, dp_align = fe._banded_levenshtein(s1, s2, fe.band_width, chain)
    max_possible = max(len(s1), len(s2)) * 2
    norm = max(0, dp_score) / max_possible
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return norm, dp_score, elapsed_ms, seeds, dp_align

norm, raw_dp, elapsed_ms, seeds, dp_align = compute_details(i, j)

# Display metrics
c1, c2, c3 = st.columns(3)
c1.metric("Similarity", f"{norm:.4f}")
c2.metric("Raw DP score", f"{raw_dp}")
c3.metric("Compute time", f"{elapsed_ms:.2f} ms")

# Detailed outputs
with st.expander("Show raw token sequences"):
    a, b = st.columns(2)
    a.subheader(name_i); a.write(scanpaths[i])
    b.subheader(name_j); b.write(scanpaths[j])

with st.expander("Greedy seed chaining output"):
    st.write(seeds)

with st.expander("DP alignment (positions)"):
    st.write(dp_align)
# streamlit run dashboard.py