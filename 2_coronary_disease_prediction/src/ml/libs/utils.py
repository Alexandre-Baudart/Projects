import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score
)

def metric_config(metric: str):
    if metric == "acc":
        return accuracy_score, "Accuracy"
    elif metric == "precision":
        return precision_score, "Precision"
    elif metric == "recall":
        return recall_score, "Recall"
    elif metric == "f1-score":
        return f1_score, "F1-Score"
    if metric == "auc":
        return roc_auc_score, "ROC-AUC"
    elif metric == "pr_auc":
        return average_precision_score, "PR-AUC"
    else:
        raise ValueError(f"Unknown metric: {metric}")

def display_confusion_matrix(conf_matrix: np.ndarray, names_list: list | None = None, cm_save_name: str | None = None) -> None :
    df_cm = DataFrame(conf_matrix, index=names_list, columns=names_list)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", linewidths=0.5, ax=ax)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    if cm_save_name : plt.savefig(f"./evaluations/{cm_save_name}_conf_matrix.png")

    plt.show()

def random_init(seed: int = 42) :
    random.seed(seed)
    np.random.seed(seed)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray, mode = "clf_binary") :
    from sklearn.preprocessing import label_binarize

    if mode != "clf_binary" :
        n_classes = y_prob.shape[1]
        y_onehot = label_binarize(y_true, classes=np.arange(n_classes))

        return np.mean(np.sum((y_prob - y_onehot) ** 2))
    else :
        return np.mean((y_prob - y_true) ** 2)

def ECE(y_true, y_prob, n_bins: int = 10) :
    # Obtenir les probas moyennes par bin et les fréquences réelles
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Retrouver la distribution des effectifs par bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1

    ece = 0.0
    n_samples = len(y_true)

    # Calcul de la somme pondérée des écarts abs(vrai - prédit)
    for i in range(n_bins) :
        # Sélection des éléments appartenant au bin i
        bin_mask = (bin_assignments == i)
        bin_size = np.sum(bin_mask)

        if bin_size > 0 :
            # Calculer la moyenne locale pour bin spécifique
            local_prob_true = np.mean(y_true[bin_mask])
            local_prob_pred = np.mean(y_prob[bin_mask])

            # Formule de l'ECE : (taille_bin / total) * |vrai - prédit|
            ece += (bin_size / n_samples) * np.abs(local_prob_pred - local_prob_true)

    return ece

def get_cpu_available() -> int:
    return os.cpu_count() - 2

def split_set(X, y, train_size: float = 0.80) :
    return train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
