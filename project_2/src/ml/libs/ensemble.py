import numpy as np
from typing import final
from sklearn.metrics import confusion_matrix, classification_report
from ..libs.utils import metric_config, display_confusion_matrix

class Ensemble :
    def __init__(self, X, y, models: list | None = None) :
        self.models = models

        self.X = X
        self.y = y


    @final
    def vote(self, return_probs: bool = False) :
        all_y_pred = []

        if self.models is None :
            raise ValueError("Models must be provided before voting.")

        for model in self.models :
            y_pred = model.predict(self.X, return_probs=return_probs)
            all_y_pred.append(y_pred)

        y_pred_ensemble = np.column_stack(all_y_pred)

        if not return_probs :
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=1,
                arr=y_pred_ensemble
            )
        else :
            return np.apply_along_axis(
                lambda x: np.mean(x),
                axis=1,
                arr=y_pred_ensemble
            )


    @final
    def _score(self, y_true, y_pred, metric: str = "acc") :
        score, metric_name  = metric_config(metric)
        print(f"{metric_name} : {score(y_true, y_pred):.4f}")


    @final
    def test(self, metrics: list | None = None, conf_matrix: bool = False, cm_save_name: str | None = None, clf_report: bool = False, cr_save_name: str = None) :
        print("\n=== Ensemble Test ===\n")

        y_pred = self.vote()
        y_prob = self.vote(return_probs=True)

        if metrics is None : metrics = ["acc"]

        for metric in metrics :
            if metric in ["auc", "pr_auc"] :
                self._score(y_true=self.y, y_pred=y_prob, metric=metric)
            else :
                self._score(y_true=self.y, y_pred=y_pred, metric=metric)

        if conf_matrix :
            cm = confusion_matrix(self.y, y_pred)
            display_confusion_matrix(conf_matrix=cm, cm_save_name=cm_save_name)

        if clf_report :
            cr = classification_report(self.y, y_pred)
            print(cr)


