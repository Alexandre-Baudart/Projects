import numpy as np
import os
import json
from typing import final

from sklearn.metrics import classification_report, confusion_matrix
from abc import ABC, abstractmethod

from .utils import metric_config
from analysis_utils import display_conf_matrix
from project_utils import plot_roc_curve, plot_pr_auc_curve


class Base(ABC) :
    def __init__(self, params: dict | None = None) :
        self.params = params

        self.model = None
        self.n_features = None
        self.classes = None


    @abstractmethod
    def optimal_params_search(self, X: np.ndarray, y: np.ndarray) -> dict :
        pass


    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, metric: str = "acc") :
        pass


    @final
    def predict(self, X: np.ndarray) :
        if not self.model : return None

        return self.model.predict(X)


    @final
    def _score(self, y_true, y_pred, metric: str = "acc", show_auc_curves: bool = False) :
        if show_auc_curves :
            if metric == "auc" :
                plot_roc_curve(y_true=y_true, y_pred=y_pred)
            elif metric == "pr_auc" :
                plot_pr_auc_curve(y_true=y_true, y_pred=y_pred)

        score, metric  = metric_config(metric)
        print(f"\nScore ({metric}) : {score(y_true, y_pred):.4f}")


    @final
    def test(self, X, y, metrics: list | None = None, conf_matrix: bool = False, cm_save_root: str | None = None, clf_report: bool = False,  cr_save_root: str | None = None) :
        print("\n=== Test ===")

        y_pred = self.predict(X)

        if y_pred is not None :
            if metrics is None : metrics = ["acc"]

            for metric in metrics :
                self._score(y_true=y, y_pred=y_pred, metric=metric)

            if conf_matrix :
                cm = confusion_matrix(y, y_pred)
                display_conf_matrix(cm=cm, save_root=cm_save_root)

            if clf_report :
                print("\n=== Classification Report ===\n")
                cr = classification_report(y, y_pred)
                print(cr)

                if cr_save_root is not None :
                    cr_path = os.path.join(cr_save_root, "clf_report.json")
                    with open(cr_path, 'w') as file:
                        json.dump(cr, file, indent=4)
        else :
            raise RuntimeError("\nErreur dans l'inférence du modèle !")


    def to_onnx(self, filename: str = "model") :
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        if None not in (self.model, self.n_features) :
            initial_type = [("input", FloatTensorType([None, self.n_features]))]

            from sklearn.preprocessing import StandardScaler

            options = {
                id(self.model): {"zipmap": False},
                StandardScaler: {"div": "div_cast"}
            }

            onnx_model = convert_sklearn(self.model, initial_types=initial_type, options=options)

            with open(f"./{filename}.onnx", "wb") as f :
                f.write(onnx_model.SerializeToString())


