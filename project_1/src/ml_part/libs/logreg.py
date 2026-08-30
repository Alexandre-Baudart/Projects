import numpy as np
import optuna
import time
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from joblib import parallel_config

from .base import Base
from project_utils import split_set
from .utils import metric_config, get_cpu_available


class LogisticReg(Base) :
    def __init__(self, params: dict | None = None, scaler: str = "standard") :
        super().__init__(params)

        self.scaler = StandardScaler if scaler == "standard" else MinMaxScaler

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        score, metric_name = metric_config(metric)
        n_jobs = get_cpu_available()

        def objective(trial) :
            params = {
                'C': trial.suggest_float("C", 1e-3, 2.0, log=True),
                "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True)
            }

            verbose = kwargs.get("verbose", 1)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model = Pipeline([
                    ("scaler", self.scaler()),
                    ("logreg", LogisticRegression(
                        fit_intercept=True,
                        max_iter=1000,
                        random_state=42,
                        verbose=verbose,
                        **params
                    ))
                ])

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

        with parallel_config(backend="loky", n_jobs=n_jobs):
            try:
                study.optimize(objective, n_trials=100, timeout=600, n_jobs=-1, catch=(KeyboardInterrupt, ))
            except KeyboardInterrupt:
                pass

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"\nMean {metric_name} :", study.best_value)
        print(f"Std {metric_name} :", study.best_trial.user_attrs["std"])

        self.params = study.best_params

    def fit(self, X, y, metric:str = "acc", **kwargs) :
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimal_params_search(X_search, y_search)

        logreg_params = {
            'C': self.params.get('C', 1e-3),
            "max_iter": self.params.get("max_iter", 1000),
            "fit_intercept": True,
            "tol": self.params.get("tol", 1e-3),
            "verbose": self.params.get("verbose", 0),
            "random_state": 42,
        }

        self.model = Pipeline([
            ("scaler", self.scaler()),
            ("logreg", LogisticRegression(**logreg_params))
        ])

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


