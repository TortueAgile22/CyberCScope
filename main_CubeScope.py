import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dill
from sklearn import preprocessing
import sys

sys.path.append("_dat")
sys.path.append("_src")

import CubeScope

def plot_regimeassignment(
    ax,
    model_obj: object,
    regime_assignments: list,
    skip=[],
    line_width=8,
):
    all_rgm_num = len(model_obj.regimes)
    length = model_obj.data_len
    ax.set_title("Regime assignments")
    
    if not regime_assignments:
        return

    r = regime_assignments[0][0]
    st = regime_assignments[0][1]
    
    n_colors = max(all_rgm_num, 10)
    palette = sns.color_palette(n_colors=n_colors)

    for assign in regime_assignments[1:]:
        ed = assign[1]
        if r not in skip:
            color_idx = (r - len(skip)) % n_colors
            ax.hlines(
                y=r - len(skip),
                xmin=st,
                xmax=ed,
                color=palette[color_idx],
                linewidth=line_width,
            )
        r = assign[0]
        st = assign[1]
    
    ed = length
    color_idx = (r - len(skip)) % n_colors
    ax.hlines(
        y=r - len(skip),
        xmin=st,
        xmax=ed,
        color=palette[color_idx],
        linewidth=line_width,
    )
    ax.set_yticks(range(max(all_rgm_num - len(skip), 1)))
    ax.set_xlabel("Time steps")


def plot_anomscore(
    ax,
    model_obj: object,
    width,
    tensor,
    label_series,
    time_idx,
    label_col
):
    print("Génération du plot Anomaly Score...")
    max_len = model_obj.data_len
    gt_series = []

    unique_labels = label_series.unique()

    has_benign_str = 'BENIGN' in unique_labels
    
    for start in range(0, max_len, width):
        end = start + width
        group_mask = (tensor[time_idx] >= start) & (tensor[time_idx] < end)
        
        if not group_mask.any():
            gt_series.append(0)
            continue
            
        indices = tensor.index[group_mask]
        current_labels = label_series.iloc[indices]
        
        is_attack = False
        if has_benign_str:
            if (current_labels != 'BENIGN').any():
                is_attack = True
        else:
            try:
                numeric_labels = pd.to_numeric(current_labels, errors='coerce').fillna(0)
                if (numeric_labels != 0).any():
                    is_attack = True
            except:
                if (current_labels != 0).any():
                    is_attack = True
        
        gt_series.append(1 if is_attack else 0)

    anomscores = model_obj.anomaly_scores
    
    min_len = min(len(anomscores), len(gt_series))
    anomscores = anomscores[:min_len]
    gt_series = gt_series[:min_len]

    if not anomscores:
        print("Warning: No anomaly scores to plot.")
        return

    # Normalisation
    anomscores_min = min(anomscores)
    anomscores_max = max(anomscores)
    if anomscores_max - anomscores_min != 0:
        scaled_anomscores = [(value - anomscores_min) / (anomscores_max - anomscores_min) for value in anomscores]
    else:
        scaled_anomscores = anomscores

    repeated_anomscores = [value for value in scaled_anomscores for _ in range(width)]
    repeated_gt_series = [value for value in gt_series for _ in range(width)]
    
    # Plot
    ax.plot(repeated_anomscores, linestyle='-', color='b', linewidth=0.5, label="Anomaly Score")
    
    # Ajout des zones rouges
    x_axis = np.arange(len(repeated_gt_series))
    y_gt = np.array(repeated_gt_series)
    
    ax.fill_between(x_axis, 0, 1, where=(y_gt==1), color='red', alpha=0.3, label="Ground Truth (Attack)", transform=ax.get_xaxis_transform())
    ax.set_title("Anomaly Score (Blue) vs Ground Truth (Red Area)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Score")
    ax.grid(True)

def prepare_data_cubescope(df, categorical_cols, continuous_cols, time_col, freq, n_bins=10):
    data = df.copy()
    
    # 1. Gestion du Temps
    data[time_col] = pd.to_datetime(data[time_col])
    data[time_col] = data[time_col].dt.round(freq)
    
    le_time = preprocessing.LabelEncoder()
    data[time_col] = le_time.fit_transform(data[time_col])
    
    # 2. Discrétisation
    for col in continuous_cols:
        try:
            data[col] = pd.qcut(data[col], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            data[col] = pd.cut(data[col], bins=n_bins, labels=False)
        data[col] = data[col].astype(str)
    
    # 3. Tout devient catégoriel
    all_cat_cols = categorical_cols + continuous_cols
    
    oe = preprocessing.OrdinalEncoder()
    data[all_cat_cols] = oe.fit_transform(data[all_cat_cols].astype(str))
    data[all_cat_cols] = data[all_cat_cols].astype(int)
    
    return data, oe, all_cat_cols

def run_cubescope():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fpath", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--time_idx", type=str)
    parser.add_argument("--categorical_idxs", type=str)
    parser.add_argument("--continuous_idxs", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--freq", type=str, default="1s")
    parser.add_argument("--init_len", type=int)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--FB", type=int, default=40)
    parser.add_argument("--N_ITER", type=int, default=10)
    
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    print(f"Chargement du fichier : {args.input_fpath}")
    raw_df = pd.read_csv(args.input_fpath)
    
    print(f"Taille du dataset chargé : {len(raw_df)} lignes")
    
    if args.time_idx in raw_df.columns:
        raw_df[args.time_idx] = pd.to_datetime(raw_df[args.time_idx])
        raw_df = raw_df.sort_values(args.time_idx).reset_index(drop=True)
        print("Dataset trié par temps et index réinitialisé.")
    else:
        print("Attention: Colonne temps introuvable pour le tri.")

    cat_cols = args.categorical_idxs.split(",") if args.categorical_idxs else []
    cont_cols = args.continuous_idxs.split(",") if args.continuous_idxs else []
    
    # Préparation
    tensor_df, oe, final_cat_cols = prepare_data_cubescope(
        raw_df, cat_cols, cont_cols, args.time_idx, args.freq
    )
    
    tensor = tensor_df[[args.time_idx] + final_cat_cols]
    
    args.categorical_idxs = ",".join(final_cat_cols)
    args.alpha = 1.0 / args.k
    args.beta = 1.0 / args.k

    print("Initialisation de CubeScope...")
    cs = CubeScope.CubeScope(
        tensor,
        k=args.k,
        width=args.width,
        init_len=args.init_len,
        outputdir=args.out_dir,
        args=args,
        verbose=args.verbose
    )
    
    # --- Phase d'Entraînement ---
    print(f"Entraînement sur les données avant t={args.init_len}...")
    train_tensor = tensor[tensor[args.time_idx] < args.init_len]
    
    regime_assignments = cs.init_infer(train_tensor, n_iter=args.N_ITER)
    
    # --- Phase Online ---
    print("Démarrage du Stream Processing...")
    max_time = tensor[args.time_idx].max()
    cs.data_len = max_time
    
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, max_time, args.width))
    except ImportError:
        iterator = range(0, max_time, args.width)

    for i in iterator:
        current_batch = tensor[
            (tensor[args.time_idx] >= i) & (tensor[args.time_idx] < i + args.width)
        ].copy()
        current_batch[args.time_idx] -= i 
        
        shift_id = cs.infer_online(current_batch, args.alpha, args.beta, n_iter=args.N_ITER)
        
        if type(shift_id) == int:
            regime_assignments.append([shift_id, i])

    cs.rgm_update_fin()

    # --- Sauvegarde et Plots ---
    print("Sauvegarde des résultats...")
    cs.continuous_idxs = [] 
    cs.categorical_idxs = final_cat_cols
    dill.dump([cs, regime_assignments, [], oe], open(f"{args.out_dir}/result.dill", "wb"))

    if regime_assignments:
        fig, axes = plt.subplots(figsize=(15, 4))
        plot_regimeassignment(axes, cs, regime_assignments)
        fig.tight_layout()
        fig.savefig(args.out_dir + "/segmentation_results.png")

    if args.anomaly and hasattr(cs, 'anomaly_scores') and len(cs.anomaly_scores) > 0:
        fig, axes = plt.subplots(figsize=(15, 4))
        plot_anomscore(axes, cs, args.width, tensor, raw_df[args.label_col], args.time_idx, args.label_col)
        fig.tight_layout()
        fig.savefig(args.out_dir + "/anomalyscore.png")
        print("Plots générés avec succès.")

if __name__ == "__main__":
    run_cubescope()