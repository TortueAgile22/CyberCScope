# CyberCScope : Détection d'Anomalies sur Flux Réseaux (Benchmark)

Ce dépôt contient le code source de **CyberCScope**, une méthode de détection d'anomalies basée sur la factorisation de tenseurs continus, ainsi que son benchmark comparatif avec deux algorithmes de l'état de l'art : **CubeScope** et **MemStream**.

## 📋 Prérequis

* Python 3.8+
* Les librairies listées dans `requirements.txt`

Installation des dépendances :
```bash
pip install -r requirements.txt
```

**Guide d'Utilisateur** : suivez ces étapes pour reproduire les expériences sur les datasets UNSW-NB15 et CIC-IDS-2017.

1. Téléchargement des Données
Vous devez récupérer les datasets originaux et les placer dans le dossier `_dat/` :

UNSW-NB15 : Téléchargez le fichier UNSW-NB15_1.csv.

Lien officiel : [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

CIC-IDS-2017 : Téléchargez le fichier correspondant au trafic du Mercredi (Wednesday) --> Wednesday-workingHours.pcap_ISCX.csv.

Lien officiel : [CIC-IDS-2017 Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?select=Wednesday-workingHours.pcap_ISCX.csv)

2. Préparation des Données

Avant de lancer les modèles, les données brutes doivent être nettoyées et formatées.

```bash
# Pour UNSW-NB15thon3 prepare_data_unsw.py
python3 prepare_data_unsw.py

# Pour CIC-IDS-2017
python3 prepare_data_cic.py
```
Cela va créer des fichiers `*_ready.csv` dans le dossier `_dat/`.

3. Exécution du modèle : CyberCScope (Ours)

Lancez l'entraînement et la détection, puis l'évaluation.

Pour UNSW-NB15 :

```bash
run_unsw.sh
# Modifiez eval_metrics.py pour pointer vers le résultat UNSW avant de lancer
python3 eval_metrics.py
```

Pour CIC-IDS-2017 :
```bash
sh run_cic.sh
# Modifiez eval_metrics.py pour pointer vers le résultat CIC avant de lancer
python3 eval_metrics.py
```

4. Exécution du benchmark : CubeScope

Pour UNSW-NB15 :

```Bash
sh run_unsw_CubeScope.sh
# Pointez eval_metrics.py vers _out/unsw_cubescope/result.dill
python3 eval_metrics.py
```

Pour CIC-IDS-2017 :

```Bash
sh run_cic_CubeScope.sh
# Pointez eval_metrics.py vers _out/cic_cubescope/result.dill
python3 eval_metrics.py
````

5. Exécution du benchmark : MemStream

Pour UNSW-NB15 :

```Bash
sh run_unsw_MemStream.sh
# Utilisez le script d'évaluation unifié
python3 eval_metrics_memstream_unified.py --dataset unsw
```

Pour CIC-IDS-2017 :

```Bash
sh run_cic_MemStream.sh
python3 eval_metrics_memstream_unified.py --dataset cic
```

6. Récupération des Résultats

Une fois les évaluations terminées, les scores résumés (ROC AUC et PR AUC) sont sauvegardés automatiquement dans les fichiers textes suivants :

- `_out/unsw_result/metrics_summary.txt` (CyberCScope UNSW)

- `_out/cic_result/metrics_summary.txt` (CyberCScope CIC)

- `_out/unsw_cubescope/metrics_summary.txt` (CubeScope UNSW)

- `_out/cic_cubescope/metrics_summary.txt` (CubeScope CIC)

- `_out/unsw_memstream/metrics_summary.txt` (MemStream UNSW)

- `_out/cic_memstream/metrics_summary.txt` (MemStream CIC)

7. Visualisation Comparative

Pour générer le graphique comparatif final (Bar Chart) regroupant tous les modèles :

Ouvrez plot_result.py.

Mettez à jour les valeurs raw avec celles trouvées dans les fichiers metrics_summary.txt de l'étape 6.

Lancez le script :

```Bash
python3 plot_result.py
```