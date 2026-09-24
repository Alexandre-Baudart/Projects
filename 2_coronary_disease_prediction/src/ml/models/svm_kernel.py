import numpy as np
import time
import optuna
from typing import Literal

from joblib import parallel_config
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .base import Base
from ..libs.utils import get_cpu_available, metric_config, split_set

class KernelSVM(Base) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None, kernel : Literal["rbf", "poly", "sigmoid"] | None = None,
                 preprocess = None,
                 use_pca: bool = False,
                 pca_var: float = 0.95,
                 use_nystroem: bool = False,
                 n_components : int = 200):

        super().__init__(mode, params, preprocess)

        self.kernel = kernel

        self.use_pca = use_pca
        self.pca_var = pca_var
        self.pca = None

        self.use_nystroem = use_nystroem
        self.n_components = n_components
        self.nystroem = None
    
    def _set_params(self, params) :        
        if params is not None :
            if self.kernel in ["rbf", "poly", "sigmoid"]:
                self.params["gamma"] = self.params.get("gamma", "scale")
            if self.kernel == "poly":
                self.params["degree"] = self.params.get("degree", 2)
                self.params["coef0"] = self.params.get("coef0", 0)
            if self.kernel == "sigmoid":
                self.params["coef0"] = self.params.get("coef0", 0)
        else :
            self.params = {}

    def _build_pipeline(self, model = None, **kwargs) :
        steps = []

        if self.preprocess is not None :
            steps.append(("preprocess", clone(self.preprocess)))

        pca = kwargs.get("pca", None)
        nystroem = kwargs.get("nystroem", None)

        if pca is not None :
            steps.append(("pca", pca))

        if nystroem is not None :
            steps.append(("nystroem", nystroem))

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

        nn = NearestNeighbors(n_neighbors=11, p=2, metric="minkowski")

        pipe_nn = self._build_pipeline(preprocess=self.preprocess, model=nn)
        pipe_nn.fit(X)

        X_t = pipe_nn[:-1].transform(X)

        dist, _ = pipe_nn[-1].kneighbors(X_t)
        distances = dist[:, 1:]

        sigma_avg = np.mean(distances, axis=1)
        sigma = np.median(sigma_avg)
        sigma = max(sigma, 1e-8)

        def objective(trial) :
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)

            sigma_min = sigma / 32
            sigma_max = sigma * 2
            sigma_optim = trial.suggest_float("sigma", sigma_min, sigma_max, log=True)
            gamma = 1 / (2 * sigma_optim ** 2)

            degree = trial.suggest_int("degree", 2, 6)
            coef0 = trial.suggest_int("coef0", 0, 2)

            if self.kernel is None :
                kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
            else :
                kernel = self.kernel

            params = {'C': C}

            if kernel not in ["rbf", "poly", "sigmoid"] :
                raise ValueError(f"Kernel {kernel} is not available!")
            else :
                params["gamma"] = gamma

                if kernel == "poly" :
                    params["degree"] = degree
                    params["coef0"] = coef0

                if kernel == "sigmoid":
                    params["coef0"] = coef0

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model = self._build_pipeline(
                    model=SVC(
                        kernel=kernel,
                        tol=1e-2,
                        random_state=42,
                        verbose=0,
                        **params
                    ))
                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                if metric == "auc" or metric == "pr_auc":
                    preds = model.decision_function(X.iloc[valid_idx])
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

            score_fn, metric_name = metric_config(metric)

            if not self.params:
                X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
                self.optimize(X_search, y_search)

            params = self.params.copy()

            sigma = params.get("sigma", -1)
            if sigma != -1:
                params["gamma"] = 1 / (2 * sigma ** 2)
            else:
                params["gamma"] = "scale"

            params = {k: v for k, v in params.items() if k != "sigma"}

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )

            scores = []

            print("\n=== Cross Validation ===")

            for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
                model = self._build_pipeline(model=SVC(**params))

                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                if calibrate:
                    if calib_set is not None:
                        X_calib, y_calib = calib_set
                        calib_method = kwargs.get("calib_method", "sigmoid")

                        self._calibrate(X_calib, y_calib, calib_method=calib_method)

                if metric == "auc" or metric == "pr_auc":
                    preds = model.decision_function(X.iloc[valid_idx])
                else:
                    preds = model.predict(X.iloc[valid_idx])

                s = score_fn(y.iloc[valid_idx], preds)
                scores.append(s)

            mean = float(np.mean(scores))
            std = float(np.std(scores))

            print(
                f"\nCross validation results : \n\tMean ({metric_name}) : {mean:.4f} \n\tStd ({metric_name}) : {std:.4f}")

    def fit(self,
            X, y,
            metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            calib_set: tuple | list | None = None,
            decision_threshold: float = 0.5,
            **kwargs) :

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search, metric)

        params = self.params.copy()

        sigma = params.get("sigma", -1)
        if sigma != -1:
            params["gamma"] = 1 / (2 * sigma ** 2)
        else:
            params["gamma"] = "scale"

        params = {k: v for k, v in params.items() if k != "sigma"}

        if self.use_pca :
            self.pca = PCA(n_components=self.pca_var)
        if self.use_nystroem :
            self.nystroem = Nystroem(n_components=self.n_components)

        self.model_ = self._build_pipeline(
            model=SVC(
                tol=1e-2,
                probability=True,
                random_state=42,
                verbose=0,
                **params
            ),
            pca=self.pca,
            nystroem=self.nystroem
        )

        print("\n=== Training ===")

        start_time = time.time()

        self.model_.fit(X, y)

        if calibrate:
            if calib_set is not None:
                X_calib, y_calib = calib_set
                calib_method = kwargs.get("calib_method", "sigmoid")

                self._calibrate(X_calib, y_calib, calib_method=calib_method)

        if metric == "auc" or metric == "pr_auc":
            y_pred = self.predict(X, return_probs=True, threshold=decision_threshold)
        else:
            y_pred = self.predict(X, threshold=decision_threshold)

        score_fn, metric_name = metric_config(metric)
        score = self._score(y, y_pred, score_fn=score_fn, metric=metric_name)

        elapsed = time.time() - start_time

        print(f"\nTraining time : {elapsed:.4f}")

        return {
            "duration_seconds": elapsed,
            "n_train_samples": len(X),
            f"score_test ({metric})": score
        }