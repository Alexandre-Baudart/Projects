from project_utils import *
from .libs.utils import *
from .libs.logreg import LogisticReg


if __name__ == "__main__" :
    random_init(seed=42)

    df = load_csv("./data/creditcard.csv", show_info=False)
    X, y = split_data(df, target="Class")

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.80)

    # X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    # X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    logreg_params = {
        'C': 0.01,
        "max_iter": 716,
        "intercept_scaling": 0.69
    }

    logreg = LogisticReg(params=logreg_params)
    # logreg.optimal_params_search(X_train, y_train, metric="auc", n_jobs=cpu_used)

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    logreg.fit(X_train, y_train, metric="pr_auc", use_cross_validation=False)
    logreg.test(X_test, y_test, metrics=metrics, conf_matrix=True)


