import numpy as np
import optuna
import time
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

from .base import Base
from .utils import metric_config, get_used_cpu
from project_utils import split_set


class SgdSVM(Base) :
    def __init__(self, params: dict | None = None, scaler: str = "standard", use_pca: bool = False, pca_var: float = 0.95) :
        super().__init__(params)

        self.scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()

        self.use_pca = use_pca
        self.pca_var = pca_var
        self.pca = None


    def optimal_params_search(self, X, y, metric: str = "acc", **kwargs) -> dict :
        X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.8)
        score, metric_name = metric_config(metric)

        n_jobs, optim_n_jobs = get_used_cpu(optim_cv_used=True)

        def objective(trial) :
            params = {
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "penalty":  trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
                "loss":  trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
                "max_iter":  trial.suggest_int("max_iter", 2000, 8000, step=1000)
            }

            verbose = kwargs.get("verbose", 1)

            model = SGDClassifier(learning_rate="optimal", tol=1e-3, n_jobs=n_jobs, verbose=verbose, random_state=42, **params)
            model.fit(X_train, y_train)

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


    def fit(self, X, y, metric: str = "acc", use_cross_validation: bool = True, **kwargs) :
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        n_jobs, cv_n_jobs = get_used_cpu(optim_cv_used=use_cross_validation)

        _, metric_name = metric_config(metric)

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            params = self.optimal_params_search(X_search, y_search)
            if params :
                self.params = params

        sgd_params = {
            "learning_rate": "optimal",
            "tol": 1e-3,
            "alpha": self.params.get("alpha", 1e-3),
            "penalty": self.params.get("penalty", "l2"),
            "loss":  self.params.get("loss", "hinge"),
            "max_iter": self.params.get("max_iter", 2000),
            "n_jobs": n_jobs,
            "verbose": self.params.get("verbose", 1),
            "random_state": 42
        }

        steps = [("scaler", self.scaler)]

        if self.use_pca :
            self.pca = PCA(n_components=self.pca_var)
            steps.append(("pca", self.pca))

        steps.append(("svc", SGDClassifier(**sgd_params)))

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

