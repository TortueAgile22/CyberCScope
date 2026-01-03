#!/bin/sh

# --- CONFIGURATION CIC-IDS-2017 ---
input_fname="cic_wednesday_ready.csv"
output_dir="_out/cic_memstream"

categorical_idxs="Dst Port"
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

# MemStream : init_len = Nombre d'événements (lignes) pour l'entraînement --> Ici, il n'y a plus de notion de temps fixe
init_len=10000

echo "Lancement MEMSTREAM sur CIC-IDS..."
rm -rf "$output_dir"

python3 main_memstream.py \
    --input_fpath "_dat/$input_fname" \
    --out_dir "$output_dir" \
    --categorical_idxs "$categorical_idxs" \
    --continuous_idxs "$continuous_idxs" \
    --label_col "$label_col" \
    --init_len $init_len \
    --epochs 20 \
    --device "cpu" \
    --anomaly