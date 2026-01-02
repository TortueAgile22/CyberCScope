#!/bin/sh

# Fichier préparé par le script python
input_fname="unsw_nb15_ready.csv"
output_dir="_out/unsw_result"

# Colonnes mappées (doivent correspondre aux valeurs de COL_MAPPING dans le python)
time_idx="Timestamp"
categorical_idxs="Protocol,Dst Port"
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

# Paramètres de l'algo
# freq="1s" est adapté si le trafic est dense. Sinon "1min"
freq="1s"
k=5
width=3
init_len=10
FB=128

echo "Lancement de CyberCScope sur $input_fname..."

# On s'assure que le dossier de sortie existe (optionnel, le script python le fait souvent)
mkdir -p "$output_dir"

python3 main.py \
    --input_fpath "_dat/$input_fname" \
    --out_dir "$output_dir" \
    --time_idx "$time_idx" \
    --categorical_idxs "$categorical_idxs" \
    --continuous_idxs "$continuous_idxs" \
    --label_col "$label_col" \
    --freq "$freq" \
    --k $k \
    --width $width \
    --init_len $init_len \
    --FB $FB \
    --verbose