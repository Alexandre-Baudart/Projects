import numpy as np
import optuna
import time

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

from .base import Base
from .utils import metric_config, get_used_cpu
from project_utils import split_set


class XGBoost(Base) :
    def __init__(self, params: dict | None = None) :
        super().__init__(params)

    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.8)
        score, metric_name = metric_config(metric)

        n_jobs, optim_n_jobs = get_used_cpu(optim_cv_used=True)

        if metric == "acc" :
            eval_metric = "error"
        elif metric == "pr_auc" :
            eval_metric = "aucpr"
        else :
            eval_metric = metric

        def objective(trial) :
            params = {
                "n_estimators": 2000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "early_stopping_rounds": 50,
                "eval_metric": eval_metric,
            }

            verbosity = kwargs.get("verbosity", 1)

            model = XGBClassifier(random_state=42, n_jobs=n_jobs, verbosity=verbosity, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            if metric_name == "roc_auc" or metric_name == "average_precision" :
                pred = model.predict_proba(X_test)[:, 1]
            else :
                pred = model.predict(X_test)

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

        xgb_params = {
            "n_estimators": self.params.get("n_estimators", 2000),
            "learning_rate": self.params.get("learning_rate", 0.01),
            "max_depth": self.params.get("max_depth", 10),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.2),
            "reg_alpha": self.params.get("reg_alpha", 1e-3),
            "reg_lambda": self.params.get("reg_lambda", 1e-3),
            "n_jobs": n_jobs,
            "verbosity": self.params.get("verbosity", 1),
            "random_state": 42,
        }

        steps = [("xgb", XGBClassifier(**xgb_params))]

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


    def to_onnx(self, filename: str = "model") :
        from onnxmltools.convert.common.data_types import FloatTensorType
        from onnxmltools import convert_xgboost

        if None not in (self.model, self.n_features) :
            initial_type = [("input", FloatTensorType([None, self.n_features]))]

            onnx_model = convert_xgboost(self.model.named_steps["xgb"], initial_types=initial_type)

            with open(f"./{filename}.onnx", "wb") as f :
                f.write(onnx_model.SerializeToString())



