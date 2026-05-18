import numpy as np
import optuna
import time

from sklearn.pipeline import Pipeline

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_validate, StratifiedKFold

from .base import Base
from project_utils import split_set
from .utils import metric_config, get_used_cpu


class Isolation_Forest(Base) :
    def __init__(self, params: dict = None) :
        super().__init__(params)

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.8)
        score, metric_name = metric_config(metric)

        n_jobs, optim_n_jobs = get_used_cpu(optim_cv_used=True)

        def objective(trial) :
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                "contamination": trial.suggest_float("contamination", 0.01, 0.2),
                "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
            }

            verbose = kwargs.get("verbose", 1)

            model = IsolationForest(random_state=42, n_jobs=n_jobs, verbose=verbose, **params)
            model.fit(X_train)

            if metric_name == "roc_auc" or metric_name == "average_precision" :
                pred = -model.score_samples(X_test)
            else :
                raw_pred = model.predict(X_test)
                pred = np.where(raw_pred == -1, 1, 0)

            return score(y_test, pred)

        print("\n=== Optimization ===\n")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600, n_jobs=optim_n_jobs)

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"Meilleur {metric_name} :", study.best_value)

        return study.best_params


    def fit(self, X, y, metric: str = "acc", use_cross_validation: bool = False, **kwargs) :
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        n_jobs, cv_n_jobs = get_used_cpu(optim_cv_used=use_cross_validation)

        _, metric_name = metric_config(metric)

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            params = self.optimal_params_search(X_search, y_search)
            if params :
                self.params = params

        isol_params = {
            "n_estimators": self.params.get("n_estimators", 500),
            "contamination": self.params.get("contamination", "auto"),
            "max_samples": self.params.get("max_samples", 1.0),
            "bootstrap": self.params.get("bootstrap", False),
            "warm_start": True,
            "n_jobs": n_jobs,
            "verbose": self.params.get("verbose", 0),
            "random_state": 42,
        }

        steps = [("random_forest", IsolationForest(**isol_params))]

        self.model = Pipeline(steps)

        if use_cross_validation :
            cv_n_splits = kwargs.get("cv_n_splits", 5)

            cv = StratifiedKFold(
                n_splits=cv_n_splits,
                shuffle=True,
                random_state=42,
            )

            print("\n=== Cross Validation ===")

            results = cross_validate(
                self.model,
                X,
                y,
                cv=cv,
                scoring=metric_name,
                n_jobs=cv_n_jobs
            )

            print(f"\nCross validation results : \n\tmean ({metric_name}) : {results["test_score"].mean():.4f} \n\tstd ({metric_name}) : {results["test_score"].std():.4f}")

        else :
            print("\n=== Training ===")

            start_time = time.time()

            self.model.fit(X, y)

            y_pred = self.predict(X)
            self._score(y, y_pred, metric=metric)

            elapsed = time.time() - start_time

            print(f"\nTraining time : {elapsed:.4f}")


