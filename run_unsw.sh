#!/bin/sh

# Fichier de 20k lignes
input_fname="unsw_20k.csv"
output_dir="_out/unsw_20k_result"

time_idx="Timestamp"
categorical_idxs="Protocol,Dst Port"
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

# --- PARAMÈTRES AJUSTÉS ---
freq="1s"
k=5
width=3         # Fenêtre fine pour capter les attaques brèves
init_len=300    # 5 min d'apprentissage (sur ~15 min totales)
FB=32          # Précision standard
N_ITER=10       # Bon compromis vitesse/qualité

echo "Nettoyage..."
rm -rf "$output_dir"

echo "Lancement sur 20k lignes..."

# Commande sécurisée
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
    --N_ITER $N_ITER \
    --verbose \
    --anomaly