import numpy as np
import optuna
import time
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .base import Base
from ..libs.utils import metric_config, get_cpu_available, split_set

class RandomForest(Base) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None,
                 preprocess=None):

        super().__init__(mode, params, preprocess)

    def optimize(self,
                 X, y,
                 metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                 **kwargs):

        score_fn, _ = metric_config(metric)
        n_jobs = get_cpu_available()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_samples": trial.suggest_float('max_samples', 0.5, 0.8)
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model = self._build_pipeline(
                    model=RandomForestClassifier(oob_score=True, random_state=42, n_jobs=n_jobs, verbose=0, **params)
                )

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

        try:
            study.optimize(objective, n_trials=100, timeout=600, n_jobs=1)
        except KeyboardInterrupt:
            pass

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"\nMean {metric} :", study.best_value)
        print(f"Std {metric} :", study.best_trial.user_attrs["std"])

        return study.best_params

    def cross_validate(self,
                       X, y,
                       metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                       n_splits: int = 5,
                       **kwargs):

        self._cross_validate(RandomForestClassifier, X, y, metric, n_splits=n_splits)

    def fit(self,
            X, y,
            metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            decision_threshold: float = 0.5,
            get_features_importance: bool = False,
            **kwargs):

        n_jobs = get_cpu_available()

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search, metric)

        if calibrate:
            X_train, X_calib, y_train, y_calib = split_set(X, y, train_size=0.9)
        else:
            X_train = X
            y_train = y

        self.model_ = self._build_pipeline(
            model=RandomForestClassifier(
                oob_score=True,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0,
                **self.params
            )
        )

        print("\n=== Training ===")

        start_time = time.perf_counter()

        self.model_.fit(X_train, y_train)

        # self.classes = self.model.named_steps["model"].classes_
        # self.n_features = len(self.model.named_steps["preprocess"].get_feature_names_out())

        if calibrate:
            calib_method = kwargs.get("calib_method", "sigmoid")
            self._calibrate(X_calib, y_calib, calib_method=calib_method)

        if metric == "auc" or metric == "pr_auc":
            y_pred = self.predict(X, return_probs=True, threshold=decision_threshold)
        else:
            y_pred = self.predict(X, threshold=decision_threshold)

        score_fn, metric_name = metric_config(metric)
        score = self._score(y, y_pred, score_fn=score_fn, metric=metric_name)

        elapsed = time.perf_counter() - start_time

        print(f"\nTraining time : {elapsed:.4f}")

        if get_features_importance:
            self._get_features_importance(top_n=kwargs.get("top_n", 10))

        return {
            "duration_seconds": elapsed,
            "n_train_samples": len(X),
            f"score_test ({metric})": score
        }

    def _get_features_importance(self, top_n: int = 10) :
        import pandas as pd

        print("\n=== Features Importance ===\n")

        preprocess_ = self.model_.named_steps["preprocess"]
        features_names = preprocess_.get_feature_names_out()
        features_names = np.array([col.replace("cat__", '').replace("num__", '') for col in features_names])

        model = self.model_.named_steps["model"]
        importances = model.feature_importances_

        feature_scores = pd.DataFrame({
            "Feature": features_names,
            "Importance": importances,
        }).head(top_n).sort_values(by="Importance", ascending=False)

        print(feature_scores)


