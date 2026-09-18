import numpy as np
import optuna
import time
from typing import Literal

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .base import Base
from ..libs.utils import metric_config, get_cpu_available, split_set

class CatBoost(Base) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None,
                 preprocess = None):

        if params is not None:
            self.cat_features = params.get("cat_features", None)
            params = {k: v for k, v in params.items() if k != "cat_features"}
        else:
            self.cat_features = None

        super().__init__(mode, params, preprocess)


    def optimize(self,
                 X,
                 y,
                 metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
                 **kwargs):

        score_fn, _ = metric_config(metric)
        n_jobs = get_cpu_available()

        if metric == "acc" :
            eval_metric = "Accuracy"
        elif metric == "auc" :
            eval_metric = "AUC"
        elif metric == "pr_auc" :
            eval_metric = "PRAUC"
        else :
            raise ValueError(f"Unknown metric : {metric}.")

        def objective(trial) :
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "depth": trial.suggest_int("depth", 2, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
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

                model = CatBoostClassifier(
                    early_stopping_rounds=50,
                    eval_metric=eval_metric,
                    thread_count=n_jobs,
                    verbose=False,
                    random_state=42,
                    save_snapshot=False,
                    cat_features=self.cat_features,
                    **params
                )

                model.fit(X_train_t, y_train, eval_set=(X_valid_t, y_valid), verbose=False)

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
                       calibrate: bool = False,
                       calib_set: tuple | list | None = None,
                       **kwargs) :

        n_jobs = get_cpu_available()
        score_fn, metric_name = metric_config(metric)

        if calibrate:
            X_train, X_calib, y_train, y_calib = split_set(X, y, train_size=0.9)
        else:
            X_train = X
            y_train = y

        if metric == "auc" :
            eval_metric = "AUC"
        elif metric == "pr_auc" :
            eval_metric = "PRAUC"
        else:
            eval_metric = "Accuracy"

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        print("\n=== Cross Validation ===")

        scores = []

        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_cv, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_cv, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            preprocess = clone(self.preprocess)

            X_cv_t = preprocess.fit_transform(X_cv, y_cv)
            X_valid_t = preprocess.transform(X_valid)

            model = CatBoostClassifier(
                early_stopping_rounds=50,
                eval_metric=eval_metric,
                thread_count=n_jobs,
                verbose=False,
                random_state=42,
                save_snapshot=False,
                cat_features=self.cat_features,
                **self.params
            )

            model.fit(X_cv_t, y_cv, eval_set=(X_valid_t, y_valid), verbose=False)

            if calibrate:
                if calib_set is not None:
                    X_calib, y_calib = calib_set
                    calib_method = kwargs.get("calib_method", "sigmoid")

                    self._calibrate(X_calib, y_calib, calib_method=calib_method)

            if metric == "auc" or metric == "pr_auc":
                y_pred = model.predict_proba(X_valid_t)[:, 1]
            else:
                y_pred = model.predict(X_valid_t)

            s = score_fn(y_valid, y_pred)
            scores.append(s)

        print(f"\nCross validation results : \n\tMean ({metric}) : {np.mean(scores):.4f} \n\tStd ({metric}) : {np.std(scores):.4f}")

    def fit(self,
            X, y,
            metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            calib_set: tuple | list | None = None,
            decision_threshold: float = 0.5,
            get_features_importance: bool = False,
            **kwargs) :

        n_jobs = get_cpu_available()

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search, metric)

        if metric == "acc" :
            eval_metric = "Accuracy"
        elif metric == "auc" :
            eval_metric = "AUC"
        elif metric == "pr_auc" :
            eval_metric = "PRAUC"
        else :
            raise ValueError(f"Unknown metric : {metric}.")

        self.model_ = self._build_pipeline(
            model=CatBoostClassifier(
                early_stopping_rounds=50,
                eval_fraction=0.2,
                eval_metric=eval_metric,
                thread_count=n_jobs,
                verbose=False,
                random_state=42,
                save_snapshot=False,
                cat_features=self.cat_features,
                **self.params
            )
        )

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
            self._get_features_importance(X, top_n=kwargs.get("top_n", 10))

        return {
            "duration_seconds": elapsed,
            "n_train_samples": len(X),
            f"score_test ({metric})": score
        }

    def _get_scores(self, X):
        return self.model_.predict(
            X,
            prediction_type="RawFormulaVal",
        )

    def _get_features_importance(self, X, top_n = 10) :
        import pandas as pd

        print("\n=== Features Importance ===\n")

        model_ = self.model_.named_steps["model"]
        preprocess_ = self.model_.named_steps["preprocess"]

        feature_names = preprocess_.get_feature_names_out()
        importances = model_.feature_importances_

        feature_scores = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).head(top_n).sort_values(by="Importance", ascending=False)

        print(feature_scores)

    def to_onnx(self, filename: str = "model") :
        from onnxmltools.convert.common.data_types import FloatTensorType
        from onnxmltools import convert_xgboost

        if self.model_ is None:
            raise RuntimeError("No model available!")

        n_features = len(self.preprocess_.get_feature_names_out())

        initial_type = [("input", FloatTensorType([None, n_features]))]

        onnx_model = convert_xgboost(self.model_, initial_types=initial_type)

        with open(f"./{filename}.onnx", "wb") as f :
            f.write(onnx_model.SerializeToString())





