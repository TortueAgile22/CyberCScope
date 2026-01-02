import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "_dat/Wednesday-workingHours.pcap_ISCX.csv"
OUTPUT_FILE = "_dat/cic_wednesday_ready.csv"

# Mapping basé sur TA liste (sans Protocol ni Timestamp)
COL_MAPPING = {
    'Destination Port': 'Dst Port',
    'Flow Duration': 'Duration',
    'Total Length of Fwd Packets': 'Src Bytes',
    'Total Length of Bwd Packets': 'Dst Bytes',
    'Label': 'Label'
}

def clean_and_convert():
    print(f"Chargement de {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Sélection et renommage
        cols_to_keep = list(COL_MAPPING.keys())
        df = df[cols_to_keep].rename(columns=COL_MAPPING)

        # --- GÉNÉRATION D'UN TIMESTAMP SYNTHÉTIQUE ---
        print("Génération de timestamps artificiels...")
        # On suppose arbitrairement 100 événements par seconde pour conserver l'ordre
        # Cela permet à l'algo de 'dérouler' le fichier comme un flux.
        # Format : Timestamp (float ou datetime). CyberCScope aime bien les dates.
        start_date = pd.Timestamp("2017-07-05 08:00:00")
        
        # On ajoute 10 millisecondes entre chaque ligne (100 lignes = 1 seconde)
        time_deltas = pd.to_timedelta(np.arange(len(df)) * 10, unit='ms')
        df['Timestamp'] = start_date + time_deltas
        
        # ---------------------------------------------

        # Nettoyage (Inf/NaN)
        df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        df.dropna(inplace=True)

        # Encodage Label (BENIGN=0, le reste=1)
        print("Encodage des labels...")
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

        print(f"Sauvegarde dans {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        print("Terminé.")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    clean_and_convert()