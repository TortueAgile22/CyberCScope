import matplotlib.pyplot as plt
import numpy as np

# Données
metrics = ['UNSW - ROC', 'UNSW - PR', 'CIC - ROC', 'CIC - PR']
x = np.arange(len(metrics))
width = 0.25

# Calcul des scores effectifs (1 - x si x < 0.5)
cyber_vals = [1-0.1009, 0.1437, 1-0.3784, 0.6777] # CyberCScope
cube_vals = [1-0.3354, 0.2080, 0.5747, 0.7860]    # CubeScope
mem_vals = [0.6517, 0.0635, 0.6232, 0.6847]      # MemStream

fig, ax = plt.subplots(figsize=(10, 6))

# Création des barres
bar1 = ax.bar(x - width, cyber_vals, width, label='CyberCScope', color='skyblue')
bar2 = ax.bar(x, cube_vals, width, label='CubeScope', color='orange')
bar3 = ax.bar(x + width, mem_vals, width, label='MemStream', color='lightgreen')

# Labels et Titre
ax.set_ylabel('Score')
ax.set_title('Performance des Modèles par Dataset (ROC Inversé & PR)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)

# Ajout des valeurs sur les barres
ax.bar_label(bar1, fmt='%.2f', padding=3)
ax.bar_label(bar2, fmt='%.2f', padding=3)
ax.bar_label(bar3, fmt='%.2f', padding=3)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("_out/comparison_plot.png", dpi=300)
plt.show()