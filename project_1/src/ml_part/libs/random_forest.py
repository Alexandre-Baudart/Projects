import numpy as np
import optuna
import time
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from joblib import parallel_config

from .base import Base
from project_utils import split_set
from .utils import metric_config, get_cpu_available


class RandomForest(Base) :
    def __init__(self, params: dict = None) :
        super().__init__(params)

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        score, metric_name = metric_config(metric)
        n_jobs = get_cpu_available()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_samples": trial.suggest_float('max_samples', 0.5, 0.8)
            }

            model = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=n_jobs, verbose=0, **params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model.fit(X[train_idx], y[train_idx])

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
        print(f"\nMean {metric_name} :", study.best_value)
        print(f"Std {metric_name} :", study.best_trial.user_attrs["std"])

        self.params = study.best_params

    def fit(self, X, y, metric: str = "acc", **kwargs) :
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        n_jobs = get_cpu_available()

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimal_params_search(X_search, y_search)

        rf_params = {
            "n_estimators": self.params.get("n_estimators", 200),
            "max_depth": self.params.get("max_depth", 5),
            "min_samples_split": self.params.get("min_samples_split", 2),
            "min_samples_leaf": self.params.get("min_samples_leaf", 2),
            "max_samples": self.params.get("max_samples", 0.8),
            "oob_score": True,
            "n_jobs": n_jobs,
            "verbose": self.params.get("verbose", 0),
            "random_state": 42,
        }

        self.model = Pipeline([("random_forest", RandomForestClassifier(**rf_params))])

        print("\n=== Training ===")

        start_time = time.time()

        self.model.fit(X, y)

        if metric == "auc" or metric == "pr_auc":
            y_pred = self.predict(X, return_probs=True)[:, 1]
        else :
            y_pred = self.predict(X)

        self._score(y, y_pred, metric=metric)

        elapsed = time.time() - start_time

        print(f"\nTraining time : {elapsed:.4f}")


