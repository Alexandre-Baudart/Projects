from project_utils import *
from .utils import *
from .svm_sgd import SgdSVM


if __name__ == "__main__" :
    random_init(seed=42)

    df = load_csv("./data/creditcard.csv", show_info=False)
    X, y = split_data(df, target="Class")

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.80)

    X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    svm_params = {
        "alpha": 0.0003,
        "penalty": "l1",
        "loss": "squared_hinge",
        "max_iter": 2000,
        "verbose": 0
    }

    svm = SgdSVM(params=svm_params)
    # svm.optimal_params_search(X_train, y_train, metric="auc", n_jobs=cpu_used, verbose=0)

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    svm.fit(X_train, y_train, metric="pr_auc", use_cross_validation=False)
    svm.test(X_test, y_test, metrics=metrics, conf_matrix=True)

