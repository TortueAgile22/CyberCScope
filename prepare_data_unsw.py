import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = "_dat/UNSW-NB15_1.csv"
OUTPUT_FILE = "_dat/unsw_nb15_ready.csv"

# Liste des 49 colonnes dans l'ordre exact du fichier NUSW-NB15_features.csv
COL_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
]

# Mapping des colonnes utiles pour CyberCScope
COL_MAPPING = {
    'Stime': 'Timestamp',      # Feature 29
    'proto': 'Protocol',       # Feature 5
    'dsport': 'Dst Port',      # Feature 4
    'dur': 'Duration',         # Feature 7
    'sbytes': 'Src Bytes',     # Feature 8
    'dbytes': 'Dst Bytes',     # Feature 9
    'Label': 'Label'           # Feature 49
}
# ---------------------

def clean_and_convert():
    print(f"Chargement de {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, header=None, names=COL_NAMES, low_memory=False)
        df.columns = df.columns.str.strip()
        df_clean = df[list(COL_MAPPING.keys())].rename(columns=COL_MAPPING)
        print("Conversion du Timestamp...")
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'], unit='s')
        df_clean['Dst Port'] = df_clean['Dst Port'].astype(str)
        print(f"Taille avant nettoyage : {len(df_clean)}")
        df_clean.dropna(inplace=True)
        print(f"Taille après nettoyage : {len(df_clean)}")

        print(f"Sauvegarde dans {OUTPUT_FILE}...")
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print("Terminé ! Le fichier est prêt pour l'analyse.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    clean_and_convert()