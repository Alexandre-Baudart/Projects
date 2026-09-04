import numpy as np
import optuna
import time

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

from .base import Base
from .utils import metric_config, get_cpu_available
from project_utils import split_set


class XGBoost(Base) :
    def __init__(self, params: dict | None = None) :
        super().__init__(params)

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        score, metric_name = metric_config(metric)
        n_jobs = get_cpu_available()

        if metric == "acc" :
            eval_metric = "error"
        elif metric == "pr_auc" :
            eval_metric = "aucpr"
        else :
            eval_metric = metric

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50, eval_metric=eval_metric, random_state=42, n_jobs=n_jobs, verbosity=0, **params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model.fit(X[train_idx], y[train_idx], eval_set=[(X[valid_idx], y[valid_idx])], verbose=False)

                if metric_name == "roc_auc" or metric_name == "average_precision":
                    preds = model.predict_proba(X[valid_idx])[:, 1]
                else:
                    preds = model.predict(X[valid_idx])

                s = score(y[valid_idx], preds)
                scores.append(s)

                trial.report(s, step=fold)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            trial.set_user_attr("std", np.std(scores))

            return np.mean(scores)

        print("\n=== Optimization ===\n")

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

        try:
            study.optimize(objective, n_trials=100, timeout=600, n_jobs=1, catch=(KeyboardInterrupt,))
        except KeyboardInterrupt:
            pass

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"Mean {metric_name} :", study.best_value)
        print(f"Std {metric_name} :", study.best_trial.user_attrs["std"])

        self.params = study.best_params

    def fit(self, X, y, metric: str = "acc", **kwargs) :
        X_train, X_valid, y_train, y_valid = split_set(X, y, train_size=0.8)

        n_jobs = get_cpu_available()

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimal_params_search(X_search, y_search)

        xgb_params = {
            "n_estimators": self.params.get("n_estimators", 2000),
            "learning_rate": self.params.get("learning_rate", 0.01),
            "max_depth": self.params.get("max_depth", 10),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.2),
            "reg_alpha": self.params.get("reg_alpha", 1e-3),
            "reg_lambda": self.params.get("reg_lambda", 1e-3),
            "verbosity": self.params.get("verbosity", 0),
            "early_stopping_rounds": self.params.get("early_stopping_rounds", 50),
        }

        if metric == "acc" :
            eval_metric = "error"
        elif metric == "pr_auc" :
            eval_metric = "aucpr"
        else :
            eval_metric = metric

        self.model = XGBClassifier(n_jobs=n_jobs, eval_metric=eval_metric, random_state=42, **xgb_params)

        print("\n=== Training ===")

        start_time = time.time()

        self.model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        # self.classes = self.model.named_steps["model"].classes_

        if metric == "auc" or metric == "pr_auc":
            y_pred = self.predict(X, return_probs=True)[:, 1]
        else :
            y_pred = self.predict(X)

        self._score(y, y_pred, metric=metric)

        elapsed = time.time() - start_time

        print(f"\nTraining time : {elapsed:.4f}")

    def to_onnx(self, filename: str = "model") :
        from onnxmltools.convert.common.data_types import FloatTensorType
        from onnxmltools import convert_xgboost

        if None not in (self.model, self.n_features) :
            initial_type = [("input", FloatTensorType([None, self.n_features]))]

            onnx_model = convert_xgboost(self.model.named_steps["xgb"], initial_types=initial_type)

            with open(f"./{filename}.onnx", "wb") as f :
                f.write(onnx_model.SerializeToString())



