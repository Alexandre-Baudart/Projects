import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import final, Literal

import torch
from torchmetrics import Accuracy, Precision, Recall, MeanSquaredError, MeanAbsoluteError, F1Score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from rich.progress import track

from ..utils import (
    criterion_config,
    metric_config,
    to_loader,
)

from ..analysis_utils import display_confusion_matrix, display_multilabel_conf_matrix


class ENNP :
    """
        Ensemble Neural Network Pipeline
    """

    def __init__(self, models, mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 ensemble_mode = "mean", batch_size: int = 64, criterion=None, metric_info: dict = None, dataloader_params: dict = None,
                 device: torch.device = torch.device("cpu")) :

        self.models = models
        self.mode = mode
        self.ensemble_mode = ensemble_mode

        self.batch_size = batch_size

        if criterion is not None :
            self.criterion = criterion
        else:
            self.criterion = criterion_config(mode)

        if metric_info is None :
            self.metric_info = {
                "metric": "acc",
                "task": "binary"
            }
        else :
            self.metric_info = metric_info

        self.score, self.metric = metric_config(mode, metric_info)

        self.use_gpu = (device != "cpu")
        self.device = device

        self.dataloader_params = dataloader_params

        self.test_loader = None


    def _targets_processing(self, targets) :
        if self.mode == "clf_binary" :
            return targets.float().view(-1, 1)

        elif self.mode == "clf_multilabel" :
            return targets.float()

        else:
            raise ValueError(f"Unknown task: {self.mode}")


    def _aggregate_logits(self, logits_models, weights=None, method: str = "mean") :
        logits_models = torch.stack(logits_models)

        if weights is not None :
            w = torch.tensor(weights, device=self.device, dtype=logits_models.dtype).view(-1, 1, 1)
            w /= w.sum() # normalisation
            return (w * logits_models).sum(dim=0)

        if method == "mean" :
            return logits_models.mean(dim=0)
        elif method == "median" :
            return logits_models.median(dim=0)
        else :
            raise ValueError(f"Unknown method: {method}")


    def _vote(self, logits_models, weights, threshold=0.5) :
        if self.mode == "clf_multiclass" :
            B, C = logits_models[0].shape

            weights = torch.tensor(weights, device=self.device)
            weights = weights / weights.sum()

            votes = torch.zeros(B, C, device=self.device)

            for w, logits in zip(weights, logits_models) :
                preds = torch.argmax(logits, dim=1)
                votes[torch.arange(B), preds] += w

            return torch.argmax(votes, dim=1)

        elif self.mode == "clf_binary" or self.mode == "clf_multilabel" :
            logits_models = torch.stack(logits_models)

            probs = torch.sigmoid(logits_models)

            if weights is not None:
                w = torch.tensor(weights, device=probs.device).view(-1, 1, 1)
                w = w / w.sum()
                probs = probs * w

            avg_probs = probs.mean(dim=0)

            return (avg_probs > threshold).float()

        return -1


    def _decode(self, logits, threshold: float = 0.5) :
        if self.mode == "clf_multiclass" :
            assert logits.dim() == 2
            return torch.argmax(logits, dim=1)

        elif self.mode == "clf_binary" or self.mode == "clf_multilabel":
            assert logits.dim() == 2
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    @final
    def ensemble_test(self, dataset_test, weights_on_models: list = None, track_description: str = "Ensemble test...") :
        y_pred = []
        y_true = []
        test_loss = []

        for m in self.models :
            m.eval()

        if self.dataloader_params is not None :
            loader = to_loader(dataset_test, batch_size=self.batch_size, **self.dataloader_params)
        else:
            loader = to_loader(dataset_test, batch_size=self.batch_size)

        with torch.no_grad() :
            for inputs, targets in track(loader, description=track_description) :
                if self.use_gpu :
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                logits_models = []

                for model in self.models :
                    logits_models.append(model(inputs))

                if self.ensemble_mode == "mean" :
                    logits = self._aggregate_logits(logits_models, weights_on_models)

                    targets = self._targets_processing(targets)
                    loss = self.criterion(logits, targets)
                    test_loss.append(loss.item())

                    pred = self._decode(logits, weights_on_models, threshold=0.5)

                elif self.ensemble_mode == "vote" :
                    pred = self._vote(logits_models, weights_on_models, threshold=0.5)

                y_true.append(targets.detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        return y_true, y_pred


    @final
    def ensemble_predict(self, dataset, weights_on_models: list = None, track_description: str = "Ensemble prediction...") :
        y_pred = []

        for m in self.models :
            m.eval()

        if self.dataloader_params is not None :
            loader = to_loader(dataset, batch_size=self.batch_size, **self.dataloader_params)
        else:
            loader = to_loader(dataset, batch_size=self.batch_size)

        with torch.no_grad() :
            for batch in track(loader, description=track_description) :
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                if self.use_gpu :
                    inputs = inputs.to(self.device)

                logits_models = []

                for model in self.models :
                    logits_models.append(model(inputs))

                if self.ensemble_mode == "mean" :
                    logits = self._aggregate_logits(logits_models, weights_on_models)
                    pred = self._decode(logits, threshold=0.5)

                elif self.ensemble_mode == "vote" :
                    pred = self._vote(logits_models, weights_on_models, threshold=0.5)

                y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred)

        return y_pred


    @final
    def evaluate_ensemble(self, dataset: list, weights_on_models: list = None, show_conf_matrix: bool = False, track_description: str = "Ensemble evaluation...") :
        y_true, y_pred = self.ensemble_test(dataset, weights_on_models, track_description)

        if "clf" in self.mode :
            # labels = torch.unique(y_true).cpu().tolist()
            metric_params = {k: v for k, v in self.metric_info.items() if k != "metric"}

            if self.mode == "clf_binary" :
                y_true = y_true.view(-1)
                y_pred = y_pred.view(-1)

            acc = Accuracy(**metric_params)(y_true, y_pred)
            precision = Precision(**metric_params)(y_true, y_pred)
            recall = Recall(**metric_params)(y_true, y_pred)
            f1 = F1Score(**metric_params)(y_true, y_pred)

            print(f"\nEnsemble evaluation results : \n\tacc : {acc:.4f} \n\tprecision : {precision:.4f} \n\trecall : {recall:.4f} \n\tF1-score : {f1}\n")

            if show_conf_matrix :
                if self.mode == "clf_multilabel" :
                    cm = multilabel_confusion_matrix(y_true, y_pred)
                    display_multilabel_conf_matrix(cm)
                elif self.mode == "clf_binary" :
                    cm = confusion_matrix(y_true, y_pred)
                    display_confusion_matrix(cm)

        else :
            mae = MeanAbsoluteError()(y_true, y_pred)
            rmse = MeanSquaredError()(y_true, y_pred)

            print(f"\nEnsemble evaluation results ~ RMSE : {rmse:.4f} - MAE : {mae:.4f}\n")

