from project_utils import *
from .libs.utils import *
from .libs.random_forest import RandomForest
from .libs.isolation_forest import Isolation_Forest

def train_test_rf(X_tr, y_tr, X_te, y_te, optimize: bool = False) :
    rf_params = {
        "n_estimators": 366,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_samples": 0.54
    }

    rf = RandomForest(params=rf_params)

    if optimize:
        X_search, _, y_search, _ = split_set(X_train, y_train, train_size=0.3)
        rf.optimal_params_search(X_search, y_search, metric="pr_auc")

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    rf.fit(X_tr, y_tr, metric="pr_auc")
    rf.test(X_te, y_te, metrics=metrics, conf_matrix=True)

if __name__ == "__main__" :
    random_init(seed=42)

    df = load_csv("./data/creditcard.csv", show_info=False)
    X_train, X_test, y_train, y_test = split_data(df, target="Class", train_size=0.8)

    # train_test_rf(X_train, y_train, X_test, y_test, optimize=False)


