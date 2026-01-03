import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "_dat/Wednesday-workingHours.pcap_ISCX.csv"
OUTPUT_FILE = "_dat/cic_wednesday_ready.csv"

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
        
        df.columns = df.columns.str.strip()
        
        cols_to_keep = list(COL_MAPPING.keys())
        df = df[cols_to_keep].rename(columns=COL_MAPPING)

        # --- GÉNÉRATION D'UN TIMESTAMP SYNTHÉTIQUE ---
        print("Génération de timestamps artificiels...")
        # On suppose arbitrairement 100 événements pour 1 seconde pour conserver l'ordre --> influence les performances des algos temporels (CyberCScope et CubeScope)
        # Cela permet à l'algo de 'dérouler' le fichier comme un flux
        start_date = pd.Timestamp("2017-07-05 08:00:00")
        
        time_deltas = pd.to_timedelta(np.arange(len(df)) * 10, unit='ms')
        df['Timestamp'] = start_date + time_deltas
        
        # ---------------------------------------------

        df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        df.dropna(inplace=True)

        print("Encodage des labels...")
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

        print(f"Sauvegarde dans {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        print("Terminé.")

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    clean_and_convert()