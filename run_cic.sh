#!/bin/sh

input_fname="cic_wednesday_ready.csv"
output_dir="_out/cic_result"

time_idx="Timestamp"
categorical_idxs="Dst Port"             
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

# --- PARAMÈTRES AJUSTÉS ---
freq="1s"
k=5
width=3         
init_len=300
FB=40
N_ITER=5
# --------------------------

echo "Nettoyage..."
rm -rf "$output_dir"

echo "Lancement sur $input_fname..."

python3 main.py --input_fpath "_dat/$input_fname" --out_dir "$output_dir" --time_idx "$time_idx" --categorical_idxs "$categorical_idxs" --continuous_idxs "$continuous_idxs" --label_col "$label_col" --freq "$freq" --k $k --width $width --init_len $init_len --FB $FB --N_ITER $N_ITER --anomaly