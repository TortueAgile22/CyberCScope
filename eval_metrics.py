import dill
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import sys
import os
from _src.CyberCScope import CyberCScope

sys.path.append("_src")

# # --- CONFIGURATION UTILISATEUR ---
# # Pour le dataset UNSW-NB15
# RESULT_FILE = "_out/unsw_result/result.dill"   # Le fichier généré par ton run complet
# DATA_FILE = "_dat/unsw_nb15_ready.csv"         # Le CSV complet utilisé
# FREQ = "1s"                                    # Doit correspondre au paramètre --freq utilisé (ex: "1s")
# OUTPUT_METRICS_FILE = "_out/unsw_result/metrics_summary.txt"

# Pour le dataset CIC-IDS-2017
RESULT_FILE = "_out/cic_result/result.dill"   # Le fichier généré par ton run complet
DATA_FILE = "_dat/cic_wednesday_ready.csv"   # Le CSV complet utilisé
FREQ = "1s"                                  # Doit correspondre au paramètre --freq utilisé (ex: "1s")
OUTPUT_METRICS_FILE = "_out/cic_result/metrics_summary.txt"
# -------------------------------

def load_results():
    print(f"--- Chargement des résultats : {RESULT_FILE} ---")
    if not os.path.exists(RESULT_FILE):
        print(f"ERREUR: Le fichier {RESULT_FILE} n'existe pas.")
        sys.exit(1)
        
    with open(RESULT_FILE, "rb") as f:
        # Structure sauvegardée dans main.py : [ccs, regime_assignments, times, oe]
        data = dill.load(f)
        ccs_model = data[0]
        return ccs_model

def load_data():
    print(f"--- Chargement des données : {DATA_FILE} ---")
    if not os.path.exists(DATA_FILE):
        print(f"ERREUR: Le fichier {DATA_FILE} n'existe pas.")
        sys.exit(1)
    
    # On ne charge que le Timestamp et le Label pour économiser la RAM
    try:
        df = pd.read_csv(DATA_FILE, usecols=['Timestamp', 'Label'])
    except ValueError:
        # Fallback si les noms de colonnes diffèrent légèrement (ex: espaces)
        df = pd.read_csv(DATA_FILE)
        df = df[['Timestamp', 'Label']]
        
    # Conversion temporelle
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def calculate_metrics():
    # 1. Charger le modèle et les scores
    model = load_results()
    anomaly_scores = np.array(model.anomaly_scores)
    
    # Récupération automatique des paramètres du run
    width = model.width
    init_len = model.init_len
    
    print(f"Paramètres récupérés du modèle -> Width: {width}, Init_len: {init_len}")
    print(f"Nombre de scores d'anomalie calculés : {len(anomaly_scores)}")

    # 2. Charger la vérité terrain (Ground Truth)
    df = load_data()
    
    # 3. Alignement Temporel
    # L'algo transforme le temps en index entiers (0, 1, 2...) selon la fréquence FREQ
    # On reproduit cette logique pour aligner les labels.
    
    print("Alignement des données temporelles...")
    # On arrondit le temps selon la fréquence (ex: à la seconde près)
    df['time_bucket'] = df['Timestamp'].dt.round(FREQ)
    
    # On trouve le temps de départ (t=0 pour l'algo)
    start_time = df['time_bucket'].min()
    
    # On convertit en index entier (nombre de 'FREQ' écoulés depuis le début)
    # Pour '1s', c'est simplement la différence en secondes.
    time_delta = (df['time_bucket'] - start_time).dt.total_seconds()
    
    # Si FREQ est différent de 1s (ex: 10s), il faut diviser. 
    # Ici on suppose 1s. Si c'est 10s, faudrait diviser par 10.
    freq_seconds = pd.Timedelta(FREQ).total_seconds()
    df['time_idx'] = (time_delta / freq_seconds).astype(int)

    # 4. Construction du vecteur y_true
    # L'algo a généré un score pour chaque fenêtre [t, t+width]
    # Les fenêtres commencent après la période d'initialisation (init_len)
    
    y_true = []
    valid_scores = []
    
    # L'algo Online commence théoriquement à l'index `init_len` 
    # et avance par pas de `width`.
    
    current_idx = init_len
    
    # On parcourt chaque score généré par le modèle
    for score in anomaly_scores:
        # Définition de la fenêtre temporelle couverte par ce score
        window_start = current_idx
        window_end = current_idx + width
        
        # On regarde s'il y a une attaque dans cette fenêtre dans les données réelles
        # On filtre les lignes dont le time_idx est dans [start, end[
        mask = (df['time_idx'] >= window_start) & (df['time_idx'] < window_end)
        window_labels = df.loc[mask, 'Label']
        
        if len(window_labels) > 0:
            # Si au moins un paquet est malveillant, la fenêtre est considérée comme attaque
            is_attack = 1 if (window_labels == 1).any() else 0
            y_true.append(is_attack)
            valid_scores.append(score)
        else:
            # Si pas de données dans cette fenêtre (trou dans le flux), on ignore ce score
            pass
            
        current_idx += width

    y_true = np.array(y_true)
    y_scores = np.array(valid_scores)
    
    print(f"Fenêtres alignées et évaluées : {len(y_true)}")

    # 5. Calcul des métriques
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
    
    # Matrice de confusion (avec un seuil dynamique basé sur les percentiles)
    # On considère le top 5% des scores comme des anomalies pour l'exemple
    threshold = np.percentile(y_scores, 95) 
    y_pred = (y_scores > threshold).astype(int)
    
    print(f"\nRapport de classification (Seuil au 95ème percentile = {threshold:.4f}):")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attaque']))
    
    # Sauvegarde des métriques dans un petit fichier texte
    with open(OUTPUT_METRICS_FILE, "w") as f:
        f.write(f"ROC AUC: {roc_auc}\nPR AUC: {pr_auc}\n")
    print(f"Métriques sauvegardées dans {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    calculate_metrics()