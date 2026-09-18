import numpy as np
import optuna
import time
from typing import Literal

from joblib import parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .base import Base
from ..libs.utils import metric_config, get_cpu_available, split_set

class LogisticReg(Base) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None,
                 preprocess=None):

        super().__init__(mode, params, preprocess)

    def _build_pipeline(self, model = None, **kwargs) :
        steps = []

        if self.preprocess is not None : steps.append(("preprocess", clone(self.preprocess)))

        selection = kwargs.get("selection", None)
        if selection is not None : steps.append(("selection", clone(selection)))

        if model is not None :
            steps.append(("model", model))
        else :
            steps.append(("model", self.model_))

        return Pipeline(steps)

    def optimize(self,
                 X, y,
                 metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                 **kwargs):

        score_fn, _ = metric_config(metric)
        n_jobs = get_cpu_available()

        def objective(trial):
            params = {
                'C': trial.suggest_float("C", 1e-3, 2.0, log=True),
                "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True)
            }

            verbose = kwargs.get("verbose", 1)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model = self._build_pipeline(
                    model=LogisticRegression(
                        fit_intercept=True,
                        max_iter=1000,
                        random_state=42,
                        verbose=verbose,
                        **params
                ))
                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                if metric == "auc" or metric == "pr_auc":
                    preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
                else:
                    preds = model.predict(X.iloc[valid_idx])

                s = score_fn(y.iloc[valid_idx], preds)
                scores.append(s)

                trial.report(s, step=fold)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            trial.set_user_attr("std", np.std(scores))

            return float(np.mean(scores))

        print("\n=== Optimization ===\n")

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

        with parallel_config(backend="loky", n_jobs=n_jobs):
            try:
                study.optimize(objective, n_trials=100, timeout=600, n_jobs=-1)
            except KeyboardInterrupt:
                pass

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"\nMean {metric} :", study.best_value)
        print(f"Std {metric} :", study.best_trial.user_attrs["std"])

        self.params = study.best_params

        return study.best_params

    def cross_validate(self,
                       X, y,
                       metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                       n_splits: int = 5,
                       calibrate: bool = False,
                       calib_set: tuple | list | None = None,
                       **kwargs):

        self._cross_validate(LogisticRegression, X, y, metric, n_splits=n_splits, calibrate=calibrate, calib_set=calib_set, **kwargs)

    def fit(self,
            X, y,
            metric:Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            calib_set: tuple | list | None = None,
            decision_threshold: float = 0.5,
            get_features_importance: bool = False,
            **kwargs):

        if not self.params:
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search)

        logreg_params = {
            'C': self.params.get('C', 1e-3),
            "tol": self.params.get("tol", 1e-3),
        }

        self.model_ = self._build_pipeline(model=LogisticRegression(max_iter=1000, fit_intercept=True, verbose=0, random_state=42, **logreg_params))

        print("\n=== Training ===")

        start_time = time.time()

        self.model_.fit(X, y)

        # self.classes = self.model.named_steps["model"].classes_
        # self.n_features = len(self.model.named_steps["preprocess"].get_feature_names_out())

        if calibrate:
            if calib_set is not None:
                X_calib, y_calib = calib_set
                calib_method = kwargs.get("calib_method", "sigmoid")

                self._calibrate(X_calib, y_calib, calib_method=calib_method)

        if metric == "auc" or metric == "pr_auc":
            y_pred = self.predict(X, return_probs=True, threshold=decision_threshold)
        else :
            y_pred = self.predict(X, threshold=decision_threshold)

        score_fn, metric_name = metric_config(metric)
        score = self._score(y, y_pred, score_fn=score_fn, metric=metric_name)

        elapsed = time.time() - start_time

        print(f"\nTraining time : {elapsed:.4f}")

        if get_features_importance:
            self._get_features_importance(top_n=kwargs.get("top_n", 10))

        return {
            "duration_seconds": elapsed,
            "n_train_samples": len(X),
            f"score_test ({metric})": score
        }

    def _get_features_importance(self, top_n: int = 5) :
        import pandas as pd

        print("\n=== Features Importance ===\n")

        features_names = self.model_.named_steps["preprocess"].get_feature_names_out()
        features_names = np.array([col.replace("cat__", '').replace("num__", '') for col in features_names])

        model_ = self.model_["model"]

        importance = pd.DataFrame({
            "feature": features_names,
            "importance": np.abs(model_.coef_[0]),
            "coefficient": model_.coef_[0],
        })

        # Sign of coefficient is important: abs(coef_) gives the force of the influence
        # whereas the original coefficient notices the direction.
        # coef > 0 -> push towards the positive class
        # coef < 0 -> push towards the negative class

        feature_importance = importance.head(top_n).sort_values(by="importance", ascending=False)

        print(feature_importance)


