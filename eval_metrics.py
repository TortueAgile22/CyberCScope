import dill
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import sys
import os

sys.path.append("_src")

# # --- CONFIGURATION UTILISATEUR ---
# # Pour le dataset UNSW-NB15 lancé avec CyberCScope
# RESULT_FILE = "_out/unsw_result/result.dill"  
# DATA_FILE = "_dat/unsw_nb15_ready.csv"         
# FREQ = "1s"                                   
# OUTPUT_METRICS_FILE = "_out/unsw_result/metrics_summary.txt"

# # Pour le dataset CIC-IDS-2017 lancé avec CyberCScope
# RESULT_FILE = "_out/cic_result/result.dill"   
# DATA_FILE = "_dat/cic_wednesday_ready.csv"  
# FREQ = "1s"                                  
# OUTPUT_METRICS_FILE = "_out/cic_result/metrics_summary.txt"

# # Pour le dataset UNSW-NB15 lancé avec CubeScope
# RESULT_FILE = "_out/unsw_cubescope/result.dill"
# DATA_FILE = "_dat/unsw_nb15_ready.csv"
# FREQ = "1s"
# OUTPUT_METRICS_FILE = "_out/unsw_cubescope/metrics_summary.txt"

# # Pour le dataset CIC-IDS-2017 lancé avec CubeScope
# RESULT_FILE = "_out/cic_cubescope/result.dill"
# DATA_FILE = "_dat/cic_wednesday_ready.csv"
# FREQ = "1s"
# OUTPUT_METRICS_FILE = "_out/cic_cubescope/metrics_summary.txt"

# # Pour le dataset UNSW-NB15 lancé avec MemStream
# RESULT_FILE = "_out/unsw_memstream/result.dill"
# DATA_FILE = "_dat/unsw_nb15_ready.csv"
# FREQ = "1s"
# OUTPUT_METRICS_FILE = "_out/unsw_memstream/metrics_summary.txt"

# Pour le dataset CIC-IDS-2017 lancé avec MemStream
RESULT_FILE = "_out/cic_memstream/result.dill"
DATA_FILE = "_dat/cic_wednesday_ready.csv"
FREQ = "1s"
OUTPUT_METRICS_FILE = "_out/cic_memstream/metrics_summary.txt"
# -------------------------------

def load_results():
    print(f"--- Chargement des résultats : {RESULT_FILE} ---")
    if not os.path.exists(RESULT_FILE):
        print(f"ERREUR: Le fichier {RESULT_FILE} n'existe pas.")
        sys.exit(1)
        
    with open(RESULT_FILE, "rb") as f:
        data = dill.load(f)
        ccs_model = data[0]
        return ccs_model

def load_data():
    print(f"--- Chargement des données : {DATA_FILE} ---")
    if not os.path.exists(DATA_FILE):
        print(f"ERREUR: Le fichier {DATA_FILE} n'existe pas.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(DATA_FILE, usecols=['Timestamp', 'Label'])
    except ValueError:
        df = pd.read_csv(DATA_FILE)
        df = df[['Timestamp', 'Label']]
        
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def calculate_metrics():
    model = load_results()
    anomaly_scores = np.array(model.anomaly_scores)

    width = model.width
    init_len = model.init_len
    
    print(f"Paramètres récupérés du modèle -> Width: {width}, Init_len: {init_len}")
    print(f"Nombre de scores d'anomalie calculés : {len(anomaly_scores)}")

    df = load_data()

    
    print("Alignement des données temporelles...")

    df['time_bucket'] = df['Timestamp'].dt.round(FREQ)
    
    start_time = df['time_bucket'].min()

    time_delta = (df['time_bucket'] - start_time).dt.total_seconds()
    

    # Ici on suppose 1s. Si c'est 10s, faudrait diviser par 10.
    freq_seconds = pd.Timedelta(FREQ).total_seconds()
    df['time_idx'] = (time_delta / freq_seconds).astype(int)
    
    y_true = []
    valid_scores = []
    
    current_idx = init_len
    
    for score in anomaly_scores:
        window_start = current_idx
        window_end = current_idx + width
        mask = (df['time_idx'] >= window_start) & (df['time_idx'] < window_end)
        window_labels = df.loc[mask, 'Label']
        
        if len(window_labels) > 0:
            is_attack = 1 if (window_labels == 1).any() else 0
            y_true.append(is_attack)
            valid_scores.append(score)
        else:
            pass
            
        current_idx += width

    y_true = np.array(y_true)
    y_scores = np.array(valid_scores)
    
    print(f"Fenêtres alignées et évaluées : {len(y_true)}")

    if len(np.unique(y_true)) < 2:
        print("\nATTENTION : Les labels ne contiennent qu'une seule classe (tout normal ou tout attaque).")
        print(f"Labels présents : {np.unique(y_true)}")
        print("Impossible de calculer l'AUC.")
        return

    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    print("\n" + "="*40)
    print(f" RÉSULTATS D'ÉVALUATION")
    print("="*40)
    print(f"ROC AUC Score : {roc_auc:.4f}")
    print(f"PR AUC Score  : {pr_auc:.4f}")
    print("="*40)
    
    threshold = np.percentile(y_scores, 95) 
    y_pred = (y_scores > threshold).astype(int)
    
    print(f"\nRapport de classification (Seuil au 95ème percentile = {threshold:.4f}):")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attaque']))
    
    with open(OUTPUT_METRICS_FILE, "w") as f:
        f.write(f"ROC AUC: {roc_auc}\nPR AUC: {pr_auc}\n")
    print(f"Métriques sauvegardées dans {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    calculate_metrics()