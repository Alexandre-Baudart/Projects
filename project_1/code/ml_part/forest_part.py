from project_utils import *
from .libs.utils import *
from .libs.random_forest import RandomForest
from .libs.isolation_forest import Isolation_Forest


def train_test_rf(X_tr, y_tr, X_te, y_te) :
    rf_params = {
        "n_estimators": 355,
        "max_depth": 7,
        "min_samples_split": 2,
        "min_samples_leaf": 8
    }

    rf = RandomForest(params=rf_params)

    # rf.optimal_params_search(X_train, y_train, metric="auc", n_jobs=cpu_used)

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    rf.fit(X_tr, y_tr, metric="pr_auc", use_cross_validation=False)
    rf.test(X_te, y_te, metrics=metrics, conf_matrix=True)


def train_test_if(X_tr, y_tr, X_te, y_te) :
    isol_params = {
        "n_estimators": 930,
        "contamination": 0.09,
        "max_samples": 0.67,
        "bootstrap": False,
    }

    isol_f = Isolation_Forest(params=isol_params)

    # isol_f.optimal_params_search(X_tr, y_tr, metric="pr_auc")

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    isol_f.fit(X_tr, y_tr, metric="pr_auc", use_cross_validation=False)
    isol_f.test(X_te, y_te, metrics=metrics, conf_matrix=True)


if __name__ == "__main__" :
    random_init(seed=42)

    df = load_csv("./data/creditcard.csv", show_info=False)
    X, y = split_data(df, target="Class")

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.80)

    # X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    # X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    train_test_if(X_train, y_train, X_test, y_test)


