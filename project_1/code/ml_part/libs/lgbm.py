import numpy as np
import optuna
import time
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from .base import Base
from .utils import metric_config, get_used_cpu
from project_utils import split_set


class LGBM(Base) :
    def __init__(self, params: dict | None = None) :
        super().__init__(params)

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.8)
        score, name = metric_config(metric)

        n_jobs, optim_n_jobs = get_used_cpu(optim_cv_used=True)

        def objective(trial) :
            params = {
                "n_estimators": 2000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 100, 1000, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 250),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            }

            verbosity = kwargs.get("verbosity", 1)

            model = LGBMClassifier(random_state=42, n_jobs=n_jobs, verbosity=verbosity, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(50), log_evaluation(0)]
            )

            if metric == "auc" :
                pred = model.predict_proba(X_test)[:, 1]
            else :
                pred = model.predict(X_test)

            return score(y_test, pred)

        print("\n=== Optimization ===\n")

        n_jobs = kwargs.get("n_jobs", 1)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600, n_jobs=optim_n_jobs)

        print("\nParamètres optimaux trouvés :", study.best_params)
        print(f"Meilleur {name} :", study.best_value)

        return study.best_params


    def fit(self, X, y, metric: str = "acc") :
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        n_jobs, optim_n_jobs = get_used_cpu(optim_cv_used=True)

        X_train, X_valid, y_train, y_valid = split_set(X, y, train_size=0.8)

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            params = self.optimal_params_search(X_search, y_search)
            if params :
                self.params = params

        lgb_params = {
            "n_estimators": self.params.get("n_estimators", 2000),
            "learning_rate": self.params.get("learning_rate", 0.01),
            "num_leaves": self.params.get("num_leaves", 31),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.2),
            "reg_alpha": self.params.get("reg_alpha", 1e-3),
            "reg_lambda": self.params.get("reg_lambda", 1e-3),
            "verbosity": self.params.get("verbosity", 1),
            "n_jobs": n_jobs,
            "random_state": 42,
        }

        steps = [("lgb", LGBMClassifier(**lgb_params))]

        self.model = Pipeline(steps)

        print("\n=== Training ===")

        start_time = time.time()

        self.model.fit(X_train, y_train, lgb__eval_set=[(X_valid, y_valid)], lgb__callbacks=[early_stopping(50), log_evaluation(0)])
        # print("\nPipeline final :", self.model)

        elapsed = time.time() - start_time

        y_pred = self.predict(X)
        self._score(y, y_pred, metric=metric)

        print("\nTraining time : ", elapsed)


