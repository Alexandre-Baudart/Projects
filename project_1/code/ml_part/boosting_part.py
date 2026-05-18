from project_utils import *
from .libs.xgb import XGBoost
from .libs.lgbm import LGBM


def xgb_train_test(X_tr, y_tr, X_te, y_te)  :
    xgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 8,
        "subsample": 0.99,
        "colsample_bytree": 0.88,
        "reg_alpha": 3.60,
        "reg_lambda": 0.04
    }

    xgb = XGBoost(params=xgb_params)

    # xgb.optimal_params_search(X_train, y_train, metric="auc", n_jobs=cpu_used)

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    xgb.fit(X_tr, y_tr, metric="pr_auc", use_cross_validation=False)
    xgb.test(X_te, y_te, metrics=metrics, conf_matrix=True)

    # xgb.to_onnx("xgb")


def lgb_train_test(X_tr, y_tr, X_te, y_te) :
    lgb_params = {
        "n_estimators": 2000,
        "learning_rate": 0.003,
        "scale_pos_weight": 523,
        "max_depth": 27,
        "min_child_weight": 6,
        "subsample": 0.92,
        "colsample_bytree": 0.83,
        "reg_alpha": 0.001,
        "reg_lambda": 1.50,
        "verbosity": -1
    }

    lgb = LGBM(params=lgb_params)

    # lgb.optimal_params_search(X_train, y_train, metric="auc", n_jobs=cpu_used, verbosity=-1)

    metrics = ["precision", "recall", "f1", "auc", "pr_auc"]

    lgb.fit(X_tr, y_tr, metric="pr_auc")
    lgb.test(X_te, y_te, metrics=metrics, conf_matrix=True)


if __name__ == "__main__" :
    df = load_csv("./data/creditcard.csv", show_info=False)
    X, y = split_data(df, target="Class")

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.80)

    # X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]

    xgb_train_test(X_train, y_train, X_test, y_test)