from project_utils import *
from .libs.xgb import XGBoost
from .libs.lgbm import LGBM

def xgb_train_test(X_tr, y_tr, X_te, y_te, optimize: bool = True)  :
    xgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.06,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.97,
        "reg_alpha": 0.34,
        "reg_lambda": 1.84
    }

    xgb = XGBoost(params=xgb_params)

    if optimize:
        X_search, _, y_search, _ = split_set(X_train, y_train, train_size=0.3)
        xgb.optimal_params_search(X_search, y_search, metric="pr_auc")

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    xgb.fit(X_tr, y_tr, metric="pr_auc")
    xgb.test(X_te, y_te, metrics=metrics, conf_matrix=True, show_auc_curves=True)

if __name__ == "__main__" :
    df = load_csv("./data/creditcard.csv", show_info=False)
    X_train, X_test, y_train, y_test = split_data(df, target="Class", train_size=0.8)

    # xgb_train_test(X_train, y_train, X_test, y_test, optimize=False)