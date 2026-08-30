import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score


def metric_config(metric: str = "acc") :
    if metric == "auc" :
        return roc_auc_score, "roc_auc"
    elif metric == "pr_auc" :
        return average_precision_score, "average_precision"
    elif metric == "precision" :
        return precision_score, "precision"
    elif metric == "recall" :
        return recall_score, "recall"
    elif metric == "f1" :
        return f1_score, "f1"
    else :
        return accuracy_score, "accuracy"


def get_cpu_available() :
    return os.cpu_count() - 2


def random_init(seed: int = 42) :
    random.seed(seed)
    np.random.seed(seed)
