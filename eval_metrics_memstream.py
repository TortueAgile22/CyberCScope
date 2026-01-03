import dill
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import sys
import os
import argparse

CONFIG = {
    'unsw': {
        'result': "_out/unsw_memstream/result.dill",
        'data': "_dat/unsw_nb15_ready.csv"
    },
    'cic': {
        'result': "_out/cic_memstream/result.dill",
        'data': "_dat/cic_wednesday_ready.csv"
    }
}

def load_results(result_file):
    print(f"--- Chargement des résultats : {result_file} ---")
    if not os.path.exists(result_file):
        print(f"ERREUR: Le fichier {result_file} n'existe pas.")
        print("Vérifie que tu as bien lancé le run correspondant.")
        sys.exit(1)
        
    with open(result_file, "rb") as f:
        data = dill.load(f)
        model_wrapper = data[0]
        return model_wrapper

def load_data(data_file):
    print(f"--- Chargement des données : {data_file} ---")
    if not os.path.exists(data_file):
        print(f"ERREUR: Le fichier de données {data_file} est introuvable.")
        sys.exit(1)

    try:
        df = pd.read_csv(data_file, usecols=['Label'], low_memory=False)
    except ValueError:
        print("Tentative de lecture complète (noms de colonnes divergents)...")
        df = pd.read_csv(data_file, low_memory=False)
        if 'Label' not in df.columns:
            print(f"ERREUR CRITIQUE: Colonne 'Label' introuvable dans {data_file}")
            sys.exit(1)
        df = df[['Label']]
    return df

def calculate_metrics(dataset_name):
    paths = CONFIG[dataset_name]
    
    # 1. Charger les scores
    model = load_results(paths['result'])
    y_scores = np.array(model.anomaly_scores)
    init_len = model.init_len
    
    print(f"Scores chargés : {len(y_scores)}")

    # 2. Charger la vérité terrain
    df = load_data(paths['data'])
    y_true_all = df['Label'].values
    
    print(f"Labels chargés : {len(y_true_all)}")

    # 3. Alignement
    min_len = min(len(y_scores), len(y_true_all))
    
    if min_len == 0:
        print("ERREUR: Aucun alignement possible (longueur 0).")
        return

    y_scores = y_scores[:min_len]
    y_true_all = y_true_all[:min_len]
    
    # 4. Exclusion de la phase d'apprentissage
    if min_len > init_len:
        y_scores_eval = y_scores[init_len:]
        y_true_eval = y_true_all[init_len:]
    else:
        print("Attention : Dataset plus petit que la période d'init ! Evaluation sur tout.")
        y_scores_eval = y_scores
        y_true_eval = y_true_all

    print(f"Nombre d'événements évalués (Test set) : {len(y_true_eval)}")

    # 5. Calcul Metrics
    unique_labels = np.unique(y_true_eval)
    if len(unique_labels) < 2:
        print(f"ERREUR: Une seule classe présente ({unique_labels}). Pas d'AUC possible.")
        return

    roc_auc = roc_auc_score(y_true_eval, y_scores_eval)
    pr_auc = average_precision_score(y_true_eval, y_scores_eval)
    
    print("\n" + "="*50)
    print(f" RÉSULTATS MEMSTREAM -> DATASET : {dataset_name.upper()}")
    print("="*50)
    print(f"ROC AUC Score : {roc_auc:.4f}")
    print(f"PR AUC Score  : {pr_auc:.4f}")
    print("="*50)

    # 6. Sauvegarde des métriques
    output_dir = os.path.dirname(paths['result'])
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n")
    print(f"Métriques sauvegardées dans : {summary_path}")
    
    threshold = np.percentile(y_scores_eval, 95)
    y_pred = (y_scores_eval > threshold).astype(int)
    
    print(f"\nRapport (Seuil top 5% = {threshold:.4f}):")
    print(classification_report(y_true_eval, y_pred, target_names=['Normal', 'Attaque']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation MemStream")
    parser.add_argument("--dataset", type=str, required=True, choices=['unsw', 'cic'], 
                        help="Choisir le dataset : 'unsw' ou 'cic'")
    
    args = parser.parse_args()
    
    calculate_metrics(args.dataset)