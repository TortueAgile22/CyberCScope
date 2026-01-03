#!/bin/sh

# --- CONFIGURATION UNSW-NB15 ---
input_fname="unsw_nb15_ready.csv"
output_dir="_out/unsw_memstream"
categorical_idxs="Protocol,Dst Port"
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

echo "Lancement MEMSTREAM..."
rm -rf "$output_dir"

python3 main_memstream.py \
    --input_fpath "_dat/$input_fname" \
    --out_dir "$output_dir" \
    --categorical_idxs "$categorical_idxs" \
    --continuous_idxs "$continuous_idxs" \
    --label_col "$label_col" \
    --epochs 20 \
    --device "cpu" 