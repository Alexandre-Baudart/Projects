import numpy as np
import json
from pathlib import Path
from typing import final, Literal

import os
import joblib
import skops.io as sio

from sklearn.metrics import classification_report, confusion_matrix
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.base import clone

from ..libs.utils import (
    metric_config,
    split_set,
    brier_score,
    ECE
)
from ..libs.calibration import SigmoidCalibration

class Base(ABC) :
    def __init__(self,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 params: dict | None = None,
                 preprocess = None,
                 **kwargs):

        self.mode = mode
        self.params = params or {}

        self.preprocess = preprocess
        self.preprocess_ = None

        self.model_ = None
        self.calibrator_ = None

    def set_preprocess(self, preprocess) :
        self.preprocess = preprocess

    def _build_pipeline(self, model, **kwargs):
        steps = []

        if self.preprocess is not None: steps.append(("preprocess", clone(self.preprocess)))

        if model is not None:
            steps.append(("model", model))
        else:
            steps.append(("model", self.model_))

        return Pipeline(steps)

    @abstractmethod
    def optimize(self, X, y, metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc", **kwargs) :
        pass

    @abstractmethod
    def cross_validate(self, X, y, metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc", n_splits: int = 5, **kwargs) :
        pass

    def _cross_validate(self, raw_model, X, y, metric: str, n_splits: int = 5) :
        score_fn, metric_name = metric_config(metric)

        if not self.params :
            X_search, _, y_search, _ = split_set(X, y, train_size=0.3)
            self.optimize(X_search, y_search)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        scores = []

        print("\n=== Cross Validation ===")

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            model = self._build_pipeline(model=raw_model(**self.params))

            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            if metric == "auc" or metric == "pr_auc":
                preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
            else:
                preds = model.predict(X.iloc[valid_idx])

            s = score_fn(y.iloc[valid_idx], preds)
            scores.append(s)

        mean = float(np.mean(scores))
        std = float(np.std(scores))

        print(f"\nCross validation results : \n\tMean ({metric_name}) : {mean:.4f} \n\tStd ({metric_name}) : {std:.4f}")

    @abstractmethod
    def fit(self,
            X, y,
            metric: Literal["acc", "precision", "recall", "f1-score", "auc", "pr_auc"] = "acc",
            calibrate: bool = False,
            decision_threshold: float = 0.5,
            **kwargs):

        pass

    def _get_scores(self, X):
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)

        return self.model_.predict_proba(X)[:, 1]

    def _get_probs(self, X):
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X, return_probs: bool = False, threshold: float = 0.5) :
        if not self.model_ : raise RuntimeError("No model available!")

        if not 0.0 < threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1!")

        if self.preprocess_ is not None:
            X_t = self.preprocess_.transform(X)
        else:
            X_t = X

        if self.calibrator_ is not None:
            scores = self._get_scores(X_t)
            probs = self.calibrator_.predict_proba(scores)
        else:
            probs = self._get_probs(X_t)

        """
        scores = self._get_scores(X_t)

        if return_probs :
            if self.calibrator_ is None:
                return scores

            return self.calibrator_.predict_proba(scores)

        if self.calibrator_ is None:
            probs = self._get_scores(X_t)
        else:
            scores = self._get_scores(X_t)
            probs = self.calibrator_.predict_proba(scores)
            
        """

        if return_probs:
            return probs

        return (probs >= threshold).astype(int)

    @final
    def _score(self,
               y_true, y_pred,
               score_fn, metric: Literal["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR_AUC"] = "Accuracy",
               print_score: bool = True):

        score = score_fn(y_true, y_pred)

        if print_score : print(f"\n{metric} : {score:.4f}")

        return score

    def _display_conf_matrix(self, y_true, y_pred, labels = None, class_names: list | None = None, save_root: str | None = None):
        from pandas import DataFrame
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if self.mode in ["clf_binary", "clf_multiclass"]:
            df_cm = DataFrame(cm, index=class_names, columns=class_names)
            fig, ax = plt.subplots(figsize=(12, 5))

            sns.heatmap(df_cm, annot=True, fmt="d", linewidths=0.5, ax=ax)
            plt.xlabel("Prediction")
            plt.ylabel("True")

        elif self.mode == "clf_multilabel":
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
            axes = axes.flatten()

            for i, (matrix, ax) in enumerate(zip(cm, axes)):
                sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues", cbar=False)

                if class_names is not None:
                    ax.set_title(class_names[i], fontweight="bold")

                ax.set_xlabel("Prediction")
                ax.set_ylabel("True")
        else:
            return

        plt.tight_layout()

        if save_root is not None and os.path.isdir(save_root):
                plt.savefig(os.path.join(save_root, "conf_matrix.png"))

        plt.show()

    @final
    def test(self,
             X, y,
             metrics: list | None = None,
             conf_matrix: bool = False,
             clf_report: bool = False,
             save_root: str | None = None,
             calib_eval: bool = False,
             decision_threshold: float = 0.5):

        results = {}

        print("\n=== Test ===")

        y_pred = self.predict(X, threshold=decision_threshold)
        y_prob = self.predict(X, return_probs=True, threshold=decision_threshold)

        if metrics is None: metrics = ["acc"]

        for metric in metrics:
            score_fn, metric_name = metric_config(metric)

            if metric in ["auc", "pr_auc"] :
                results[metric] = self._score(y_true=y, y_pred=y_prob, score_fn=score_fn, metric=metric_name)
            else:
                results[metric] = self._score(y_true=y, y_pred=y_pred, score_fn=score_fn, metric=metric_name)

        if calib_eval:
            results["NLL"] = log_loss(y, y_prob)
            results["Brier-Score"] = brier_score(y, y_prob)
            results["ECE"] = ECE(y, y_prob)

            print(f"\nBrier-Score : {results["Brier-Score"]:.4f}, ECE : {results["ECE"]:.4f}")

        if conf_matrix:
            self._display_conf_matrix(
                y_true=y, y_pred=y_prob,
                save_root=save_root
            )

        if clf_report:
            print("\n=== Classification Report ===\n")
            cr = classification_report(y, y_pred)
            print(cr)

            if save_root:
                with open(f"clf_report.json", "w") as file:
                    json.dump(cr, file, indent=4)

        return results

    def _calibrate(self, X, y,  calib_method: Literal["sigmoid"] = "sigmoid"):
        if calib_method == "sigmoid":
            scores = self.predict(X, return_probs=True)

            self.calibrator_ = SigmoidCalibration()
            self.calibrator_.fit(scores, y)

    def check_high_confidence_bias(self, X, y, threshold= 0.9) -> tuple :
        if self.preprocess_ is not None:
            X_t = self.preprocess_.transform(X)
        else:
            X_t = X

        y_prob = self.predict(X_t, return_probs=True)

        # Isolation of the highly confident predictions
        mask = y_prob >= threshold
        n_high_conf = np.sum(mask)

        if n_high_conf == 0:  # no prediction over the threshold
            return False

        # Calculation of the proportion of these prediction in the dataset
        # pct = n_high_conf / len(y_prob)

        # Comparison between the mean confidence and the real precision.
        avg_confidence = np.mean(y_prob[mask])
        actual_acc = np.mean(y[mask])

        # Overconfidence gap
        gap = avg_confidence - actual_acc

        # Alert threshold : if the gap is over 5% -> a recalibration is required
        if gap > 0.05:
            return True, gap
        else:
            return False, gap

    def benchmark(self, X, n_iterations: int = 100):
        import time
        from rich.progress import track

        # Warm-up
        for _ in range(10):
            self.predict(X)

        latencies = []

        # n_repeats to reduce the impact of perf_counter() cost and
        # the "system noise" (important for models with high inference speed)

        # n_repeats = 10

        print("\n=== Benchmark ===\n")

        for _ in track(
                range(n_iterations),
                description=f"Benchmarking { self.__class__.__name__ }",
            ):

            start = time.perf_counter()

            self.predict(X)

            end = time.perf_counter()

            latencies.append(end - start)

        latencies = np.asarray(latencies)

        mean_latency = latencies.mean()
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        throughput = float(len(X) / mean_latency)

        print(f"\nResults : \n\tMean latency : {mean_latency * 1000:.4f} ms \
            \n\tP50 : {p50 * 1000:.4f} ms \
            \n\tP95 : {p95 * 1000:.4f} ms \
            \n\tP99 : {p99 *1000:.4f} ms \
            \n\tThroughput : {throughput:.2f} samples/s \
        ")

        return {
            "mean_latency_ms": mean_latency,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "throughput_samples_per_second": throughput,
        }

    def to_onnx(self, filename: str = "model"):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        if self.model_ is None:
            raise RuntimeError("No model available!")

        n_features = len(self.model_.named_steps["preprocess"].get_feature_names_out())

        initial_type = [("input", FloatTensorType([None, n_features]))]

        from sklearn.preprocessing import StandardScaler

        options = {
            id(self.model_): {"zipmap": False},
            StandardScaler: {"div": "div_cast"}
        }

        onnx_model = convert_sklearn(self.model_, initial_types=initial_type, options=options)

        with open(f"./{filename}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

    def save(self, path: Path | None = None, format_="joblib"):
        if path is None:
            filename = self.model_.__class__.__name__.lower()
            root = "./runs/sandbox/ml_models"
            os.makedirs(root, exist_ok=True)

            path = os.path.join(root, f"{filename}.joblib")

        artifact = {}

        if self.preprocess_ is not None:
            artifact["preprocess"] = self.preprocess_

        if self.model_ is None:
            raise RuntimeError("No fitted model available!")

        artifact["model"] = self.model_

        if self.calibrator_ is not None:
            artifact["calibrator"] = self.calibrator_

        if format_ == "joblib":
            joblib.dump(artifact, path)
        elif format_ == "skops":
            sio.dump(artifact, path)

        print(f"\nNote: Model has been saved successfully!")

    @classmethod
    def load(cls, model_path: Path | None = None):
        if model_path is not None:
            ext = os.path.splitext(model_path)[1].lower()

            if ext == ".joblib":
                artifact = joblib.load(model_path)
            elif ext == ".skops":
                unknown_types = sio.get_untrusted_types(
                    file=model_path
                )

                artifact = sio.load(
                    model_path,
                    trusted=unknown_types
                )
            else:
                raise ValueError(f"Unknown format: {ext}")

            instance = cls()

            instance.preprocess_ = artifact.get("preprocess", None)
            instance.model_ = artifact.get("model", None)
            instance.calibrator_ = artifact.get("calibrator", None)

            print(f"\nNote: Model has been loaded with success.")
        else :
            raise ValueError("A model path must be provided.")

        return instance