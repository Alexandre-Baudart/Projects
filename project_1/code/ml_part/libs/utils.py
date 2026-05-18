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


def get_used_cpu(optim_cv_used: bool = False) :
    os_count = os.cpu_count()
    cpu_used = None
    cpu_used_cv = None

    if os_count :
        if os_count > 2 :
            if optim_cv_used :
                cpu_used_tmp = os_count - 2
                cpu_used = cpu_used_tmp // 2
                cpu_used_cv = cpu_used_tmp - cpu_used

                return cpu_used, cpu_used_cv
            else :
                cpu_used = os_count - 2
        else :
            cpu_used = os_count

        os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_used)

    return cpu_used, cpu_used_cv


def random_init(seed: int = 42) :
    random.seed(seed)
    np.random.seed(seed)
