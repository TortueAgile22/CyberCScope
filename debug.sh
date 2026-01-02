#!/bin/sh

# On utilise le mini fichier
input_fname="mini_debug.csv"
output_dir="_out/debug_result"

time_idx="Timestamp"
categorical_idxs="Protocol,Dst Port"
continuous_idxs="Duration,Src Bytes,Dst Bytes"
label_col="Label"

# --- PARAMÈTRES DEBUG (ULTRA RAPIDE) ---
freq="1s"
k=5
width=10        # Fenêtres petites (10s) car le fichier est court
init_len=60     # On apprend juste sur la première minute
FB=40           # Précision réduite (suffisant pour test)
N_ITER=1        # 1 seule itération (qualité nulle, mais test le code instantanément)

echo "Nettoyage..."
rm -rf "$output_dir"

echo "Lancement DEBUG..."

# NOTE : J'ai mis tous les arguments sur une seule ligne pour éviter les erreurs de '\'
python3 main.py --input_fpath "_dat/$input_fname" --out_dir "$output_dir" --time_idx "$time_idx" --categorical_idxs "$categorical_idxs" --continuous_idxs "$continuous_idxs" --label_col "$label_col" --freq "$freq" --k $k --width $width --init_len $init_len --FB $FB --N_ITER $N_ITER --verbose --anomaly