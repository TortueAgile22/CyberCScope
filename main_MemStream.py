import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import dill

sys.path.append("_src")

import MemStream


def plot_anomscore_memstream(ax, scores, labels, init_len):
    min_len = min(len(scores), len(labels))
    scores = np.array(scores[:min_len])
    labels = np.array(labels[:min_len])
    
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 0:
        scaled_scores = (scores - s_min) / (s_max - s_min)
    else:
        scaled_scores = scores

    ax.plot(scaled_scores, color='blue', linewidth=0.5, label="Anomaly Score", alpha=0.8)
    
    ax.axvspan(0, init_len, color='gray', alpha=0.3, label="Initialization (Train)")

    attack_indices = np.where(labels == 1)[0]

    if len(attack_indices) > 0:
        for idx in attack_indices:
            ax.axvline(x=idx, color='red', linewidth=1, alpha=0.1) 
    ax.set_title("MemStream: Anomaly Score (Blue) vs Ground Truth (Red)")
    ax.set_xlabel("Time (Events)")
    ax.set_ylabel("Normalized Score")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.5)

def run_memstream():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--categorical_idxs", type=str)
    parser.add_argument("--continuous_idxs", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--init_len", type=int, default=5000, help="Nombre d'événements pour l'entraînement initial")
    parser.add_argument("--width", type=int, default=1, help="Non utilisé par MemStream, gardé pour compatibilité")
    parser.add_argument("--freq", type=str, help="Non utilisé par MemStream")
    
    parser.add_argument("--memlen", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--anomaly", action="store_true")

    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. Préparation des données
    print("Loading data...")
    df = pd.read_csv(args.input_fpath, low_memory=False)
    
    cat_cols = args.categorical_idxs.split(",") if args.categorical_idxs else []
    cont_cols = args.continuous_idxs.split(",") if args.continuous_idxs else []
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    scaler = StandardScaler()
    if cont_cols:
        df[cont_cols] = scaler.fit_transform(df[cont_cols])
        
    feature_cols = cat_cols + cont_cols
    X = df[feature_cols].values
    y = df[args.label_col].values 
    
    # 2. Split Train (Init) / Test (Stream)
    N_INIT = args.init_len
    if N_INIT >= len(X):
        N_INIT = int(len(X) * 0.2) 
        print(f"Warning: init_len too large, adjusted to {N_INIT}")

    X_train = torch.FloatTensor(X[:N_INIT])
    X_test = torch.FloatTensor(X[N_INIT:])
    
    device = torch.device(args.device)
    
    # 3. Initialisation & Entraînement Modèle
    print("Initializing MemStream...")
    params = {
        'beta': 0.1, 
        'memory_len': args.memlen,
        'batch_size': 256,
        'lr': args.lr,
        'device': device
    }
    
    model = MemStream.MemStream(in_dim=X.shape[1], params=params).to(device) 
    
    print("Training Autoencoder (Initialization phase)...")
    model.mem_data = X_train.to(device)
    model.train_autoencoder(X_train.to(device), epochs=args.epochs) 
    
    model.initialize_memory(X_train.to(device)) 
    
    # 4. Streaming / Inference
    print("Streaming inference...")
    scores = []
    
    scores.extend([0.0] * N_INIT)
    
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i].unsqueeze(0).to(device)
            score = model(sample)
            scores.append(score.item())
            
            if i % 5000 == 0 and i > 0:
                print(f"Processed {i}/{len(X_test)} events")

    # 5. Sauvegarde et Plot
    class ResultWrapper:
        def __init__(self, scores, width, init_len):
            self.anomaly_scores = scores
            self.width = 1 
            self.init_len = init_len
            self.time_idx = "Timestamp" 

    res = ResultWrapper(scores, 1, N_INIT)
    
    print(f"Saving results to {args.out_dir}/result.dill")
    dill.dump([res, [], [], None], open(f"{args.out_dir}/result.dill", "wb"))
    
    if args.anomaly:
        fig, ax = plt.subplots(figsize=(15, 4))
        plot_anomscore_memstream(ax, scores, y, N_INIT)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/anomalyscore.png")
        print(f"Plot saved to {args.out_dir}/anomalyscore.png")

if __name__ == "__main__":
    run_memstream()