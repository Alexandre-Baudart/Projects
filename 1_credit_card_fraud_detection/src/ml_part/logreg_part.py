from project_utils import *
from .libs.utils import *
from .libs.logreg import LogisticReg
import numpy as np

if __name__ == "__main__" :
    random_init(seed=42)

    df = load_csv("./data/creditcard.csv", show_info=False)
    X_train, X_test, y_train, y_test = split_data(df, target="Class", train_size=0.8)

    logreg_params = {
        'C': 0.008,
        "tol": 1e-3,
    }

    logreg = LogisticReg(params=logreg_params)

    X_search, _, y_search, _ = split_set(X_train, y_train, train_size=0.3)
    # logreg.optimal_params_search(X_search, y_search, metric="pr_auc")

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    # logreg.fit(X_train, y_train, metric="pr_auc")
    # logreg.test(X_test, y_test, metrics=metrics, conf_matrix=True)

