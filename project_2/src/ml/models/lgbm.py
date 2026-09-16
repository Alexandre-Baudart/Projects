import numpy as np
import optuna
import time
from typing import Literal

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .base import Base
from ..libs.utils import metric_config, get_cpu_available, split_set

class LGBM(Base) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None,
                 preprocess=None):

        super().__init__(mode, params, preprocess)

        self.preprocess_ = None

    def optimize(self,
                 X, y,
                 metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                 **kwargs):

        score_fn, _ = metric_config(metric)
        n_jobs = get_cpu_available()

        def objective(trial) :
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 100, 1000, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 250),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_valid = X.iloc[valid_idx]
                y_valid = y.iloc[valid_idx]

                preprocess = clone(self.preprocess)

                X_train_t = preprocess.fit_transform(X_train, y_train)
                X_valid_t = preprocess.transform(X_valid)

                model = LGBMClassifier(
                    n_estimators=2000,
                    n_jobs=n_jobs,
                    verbosity=0,
                    random_state=42,
                    **params
                )

                model.fit(
                    X_train_t, y_train,
                    eval_set=[(X_valid_t, y_valid)],
                    callbacks=[early_stopping(50), log_evaluation(0)]
                )

                if metric == "auc" or metric == "pr_auc":
                    preds = model.predict_proba(X_valid_t)[:, 1]
                else:
                    preds = model.predict(X_valid_t)

                s = score_fn(y_valid, preds)
                scores.append(s)

                trial.report(s, step=fold)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            trial.set_user_attr("std", np.std(scores))

            return float(np.mean(scores))

        print("\n=== Optimization ===\n")

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

        try:
            study.optimize(objective, n_trials=100, timeout=600, n_jobs=1)
        except KeyboardInterrupt:
            pass

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"Mean {metric} :", study.best_value)
        print(f"Std {metric} :", study.best_trial.user_attrs["std"])

        self.params = study.best_params

        return study.best_params

    def cross_validate(self,
                       X, y,
                       metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                       n_splits: int = 5,
                       **kwargs):

        self._cross_validate(LGBMClassifier, X, y, metric, n_splits=n_splits)

    def fit(self,
            X, y,
            metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            **kwargs) :

        n_jobs = get_cpu_available()

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search, metric)

        X_train, X_valid, y_train, y_valid = split_set(X, y, train_size=0.8)

        if calibrate:
            X_train, X_calib, y_train, y_calib = split_set(X_train, y_train, train_size=0.9)

        self.preprocess_ = clone(self.preprocess)

        X_train_t = self.preprocess_.fit_transform(X_train, y_train)
        X_valid_t = self.preprocess_.transform(X_valid)

        self.model_ = LGBMClassifier(
            n_estimators=2000,
            n_jobs=n_jobs,
            verbosity=0,
            random_state=42,
            **self.params
        )

        print("\n=== Training ===")

        start_time = time.time()

        self.model_.fit(X_train_t, y_train, eval_set=[(X_valid_t, y_valid)], callbacks=[early_stopping(50), log_evaluation(0)])

        # self.classes = self.model.named_steps["model"].classes_
        # self.n_features = len(self.model.named_steps["preprocess"].get_feature_names_out())

        if calibrate:
            calib_method = kwargs.get("calib_method", "sigmoid")
            self._calibrate(X_calib, y_calib, calib_method=calib_method)

        elapsed = time.time() - start_time

        y_pred = self.predict(X)

        score_fn, metric_name = metric_config(metric)
        score = self._score(y, y_pred, score_fn=score_fn, metric=metric_name)

        print(f"\nTraining time : {elapsed}:.4f")

        return {
            "duration_seconds": elapsed,
            "n_train_samples": len(X),
            f"score_test ({metric})": score
        }


