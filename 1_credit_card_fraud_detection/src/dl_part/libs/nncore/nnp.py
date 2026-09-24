import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import final, Literal

import torch
from torchinfo import summary
import torch.nn as nn

from torch.amp import GradScaler, autocast
from timm.utils import ModelEmaV2

from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError, MeanAbsoluteError, AUROC, AveragePrecision
from sklearn.metrics import confusion_matrix

from rich.progress import track
# from tqdm import tqdm

from ..history import History
from ..utils import (
    optimizer_config,
    criterion_config,
    train_valid_loaders,
    metric_config,
    to_loader
)
from ..callbacks import TimeMeasuring, TrainResultsDisplay, BestModelCallback, CheckpointCallback, ProgressiveUnfreezingCallback
from ..analysis_utils import display_conf_matrix


class NNP :
    """
        Neural Network Pipeline
    """
    def __init__(self, model: nn.Module | None = None, mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 batch_size: int = 64, criterion = None, precision: int | str = 32, memory_format = None, metric_info: dict | None = None,
                 dataloader_params: dict | None = None, device: torch.device = torch.device("cpu")) :

        self.model = model
        self.mode = mode

        self.last_epoch = 0

        self.lr = None
        self.optimizer = None
        self.batch_size = batch_size

        self.model_ema = None

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

        self.score_fn, self.metric = metric_config(mode, metric_info)

        self.precision = precision

        if self.precision == 32 :
            self.use_amp = False
        elif self.precision in [16, "16-mixed"]:
            self.use_amp = (device.type == "cuda")
        else:
            raise ValueError("Unsupported precision")

        self.memory_format = memory_format

        self.use_gpu = (device != "cpu")
        self.device = device

        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.dataloader_params = dataloader_params
        self.is_pin_memory = self.dataloader_params.get("pin_memory", False) if self.dataloader_params is not None else False

        self.batch_proc_fn = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.callbacks = None
        self.history = History(mode=mode)


    def _batch_processing(self, inputs, targets) :
        if self.batch_proc_fn is not None :
            inputs, targets = self.batch_proc_fn(inputs, targets)

        return inputs, targets


    """
    def _targets_processing(self, targets) :
        if self.mode in ["clf_binary", "clf_multiclass"] :
            return targets.long().view(-1)

        elif self.mode in ["clf_multilabel", "reg"] :
            return targets.float()

        else:
            raise ValueError(f"Unknown task: {self.mode}")
    """

    @final
    def _compute_score(self, y_true, y_pred) :
        self.score_fn.reset()

        y_true = y_true.long()

        self.score_fn.update(y_pred, y_true)
        score = self.score_fn.compute()

        return score


    def train(self, dataset_train = None, n_epochs: int = 10, lr: float = 1e-3, optimizer = None, accumulate_grad_batches: int = 1,
              clip_grad_norm: bool = False, batch_proc_fn = None, use_stratified_split: bool = False, use_ema: bool = False, ema_decay: float = 0.999,
              unfreezing_schedule=None, callbacks: dict | None = None, save_best_model: bool = False, enable_checkpoints: bool = False, save_root: str = "./runs/") :

        if dataset_train is not None :
            self.lr = lr
            self.batch_proc_fn = batch_proc_fn

            if optimizer is None :
                self.optimizer = optimizer_config(model_or_params=self.model, optim_info=dict(type="sgd", lr=self.lr))
            else:
                self.optimizer = optimizer

            if use_ema and self.model_ema is None :
                self.model_ema = ModelEmaV2(self.model, decay=ema_decay, device=self.device)

            self.callbacks = {} if callbacks is None else callbacks
            self.callbacks["time_measuring"] = TimeMeasuring()
            self.callbacks["tr_display"] = TrainResultsDisplay()

            if unfreezing_schedule is not None :
                self.callbacks["progressive_unfreezing"] = ProgressiveUnfreezingCallback(self.model, unfreezing_schedule)

            if self.train_loader is None and self.valid_loader is None :
                self.train_loader, self.valid_loader = train_valid_loaders(dataset_train, batch_size=self.batch_size,
                                                                dataloader_params=self.dataloader_params, use_stratified_split=use_stratified_split)

            if save_best_model :
                self.callbacks["best_model"] = BestModelCallback(
                    self.model,
                    self.optimizer,
                    self.batch_size,
                    self.callbacks["scheduler"],
                    self.history,
                    self.model_ema,
                    save_root
                )

            if enable_checkpoints :
                self.callbacks["checkpoint"] = CheckpointCallback(
                    self.model,
                    self.optimizer,
                    self.batch_size,
                    self.callbacks["scheduler"],
                    self.history,
                    self.model_ema,
                    save_root
                )

            for cb in self.callbacks.values() :
                cb.on_train_begin()

            for epoch in range(self.last_epoch, n_epochs) :
                try :
                    for cb in self.callbacks.values() :
                        cb.on_epoch_begin(epoch)

                    self.model.train()
                    self.optimizer.zero_grad()

                    print(f"\nEpoch {epoch+1}/{n_epochs} : \n")

                    with torch.enable_grad() :
                        for batch, (inputs, targets) in enumerate(track(self.train_loader, description="Training...")) :
                            if self.use_gpu :
                                inputs, targets = inputs.to(self.device, memory_format=self.memory_format, non_blocking=self.is_pin_memory), targets.to(
                                    self.device, non_blocking=self.is_pin_memory)

                            with autocast(device_type=self.device.type, enabled=self.use_amp) :
                                inputs, targets = self._batch_processing(inputs, targets)

                                output = self.model(inputs)

                                if hasattr(output, "logits") :
                                    output = output.logits

                                # targets = self._targets_processing(targets)

                                loss = self.criterion(output, targets)

                            loss = loss / accumulate_grad_batches

                            if self.use_amp :
                                self.scaler.scale(loss).backward()

                                if (batch + 1) % accumulate_grad_batches == 0 :
                                    if clip_grad_norm :
                                        self.scaler.unscale_(self.optimizer)
                                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                    self.scaler.step(self.optimizer)
                                    self.scaler.update()
                                    self.optimizer.zero_grad()
                            else :
                                loss.backward()

                                if (batch + 1) % accumulate_grad_batches == 0 :
                                    if clip_grad_norm :
                                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                    self.optimizer.step()
                                    self.optimizer.zero_grad()

                            if use_ema :
                                self.model_ema.update(self.model)

                            for cb in self.callbacks.values() :
                                cb.on_batch_end(batch, logs={"loss": loss.item()})

                    y_true_train, y_pred_train, train_loss = self._validate(used_loader="train", track_description="Validation (training set)...")
                    y_true_valid, y_pred_valid, valid_loss = self._validate(used_loader="validation", track_description="Validation (validation set)...")

                    train_score = self._compute_score(y_true_train, y_pred_train)
                    valid_score = self._compute_score(y_true_valid, y_pred_valid)

                    lr = self.callbacks["scheduler"].current_lr if "scheduler" in self.callbacks else self.lr

                    for cb in self.callbacks.values() :
                        cb.on_epoch_end(epoch,
                                        logs={"train_score": train_score, "train_loss": train_loss,
                                              "valid_score": valid_score,
                                              "valid_loss": valid_loss, "metric": self.metric})

                    self.history.save(
                        dict(train_score=train_score, valid_score=valid_score, train_loss=train_loss,
                             valid_loss=valid_loss, lr=lr)
                    )

                    if any(getattr(cb, "stop_training", False) for cb in self.callbacks.values()):
                        break

                    self.last_epoch += 1

                except KeyboardInterrupt:
                    print(f"\nQuick stop at epoch {epoch+1}.")
                    break

            for cb in self.callbacks.values(): cb.on_train_end()
            
            
    def _decode(self, logits, threshold: float = 0.5, return_probs: bool = False) :
        if self.mode == "clf_multiclass":
            return torch.argmax(logits, dim=1)

        elif self.mode == "clf_binary" or self.mode == "clf_multilabel" :
            probs = torch.sigmoid(logits)

            if return_probs :
                return probs

            return (probs > threshold).float()

        elif self.mode == "reg":
            return logits.squeeze(-1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def _validate(self, used_loader: str = "validation", probs_threshold: float = 0.5, track_description: str = "Validation...") :
        y_true = []
        y_pred = []
        valid_loss = []

        ref_model = self.model if self.model_ema is None else self.model_ema.module

        ref_model.eval()

        if used_loader == "train" :
            loader = self.train_loader
        elif used_loader == "test" :
            loader = self.test_loader
        else :
            loader = self.valid_loader

        return_probs = True if self.metric in ["AUC", "PR-AUC"] else False

        with torch.no_grad() :
            for inputs, targets in track(loader, description=track_description) :
                if self.use_gpu :
                    inputs, targets = inputs.to(self.device, memory_format=self.memory_format, non_blocking=self.is_pin_memory), targets.to(self.device,
                                                non_blocking=self.is_pin_memory)

                with autocast(device_type=self.device.type, enabled=self.use_amp) :
                    output = ref_model(inputs)

                    if hasattr(output, "logits") :
                        output = output.logits

                    # targets = self._targets_processing(targets)
                    loss = self.criterion(output, targets)
                    valid_loss.append(loss.item())

                y_true.append(targets.detach().cpu())

                pred = self._decode(output, threshold=probs_threshold, return_probs=return_probs)
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        avg_valid_loss = sum(valid_loss) / len(valid_loss)

        return y_true, y_pred, avg_valid_loss


    def test(self, dataset_test, probs_threshold: float = 0.5, track_description: str = "Test...") :
        if self.dataloader_params is not None :
            self.test_loader = to_loader(dataset_test, batch_size=self.batch_size, **self.dataloader_params)
        else :
            self.test_loader = to_loader(dataset_test, batch_size=self.batch_size)

        y_true_test, y_pred_test, test_loss = self._validate(used_loader="test", probs_threshold=probs_threshold, track_description=track_description)

        test_score = self._compute_score(y_true_test, y_pred_test)
        print(f"Test results ~ Test {self.metric} : {test_score:.4f} - Test loss : {test_loss:.4f}\n")

        return y_true_test, y_pred_test


    def predict(self, dataset, probs_threshold: float = 0.5, track_description: str = "Prediction...") :
        y_pred = []

        ref_model = self.model if self.model_ema is None else self.model_ema.module
        ref_model.eval()

        if self.dataloader_params is not None :
            loader = to_loader(dataset, batch_size=self.batch_size, **self.dataloader_params)
        else:
            loader = to_loader(dataset, batch_size=self.batch_size)

        with torch.no_grad() :
            for batch in track(loader, description=track_description) :
                if isinstance(batch, (list, tuple)) :
                    inputs = batch[0]
                else :
                    inputs = batch

                if self.use_gpu :
                    inputs = inputs.to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp) :
                    output = ref_model(inputs)

                pred = self._decode(output, threshold=probs_threshold)
                y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred)

        return y_pred


    @final
    def evaluate(self, dataset, show_conf_matrix: bool = False, probs_threshold: float = 0.5, track_description: str = "Evaluation...") :
        y_true = []
        y_prob = []
        y_pred = []

        ref_model = self.model if self.model_ema is None else self.model_ema.module
        ref_model.eval()

        if self.dataloader_params is not None :
            loader = to_loader(dataset, batch_size=self.batch_size, **self.dataloader_params)
        else:
            loader = to_loader(dataset, batch_size=self.batch_size)

        with torch.no_grad() :
            for inputs, targets in track(loader, description=track_description) :
                if self.use_gpu :
                    inputs, targets = inputs.to(self.device, memory_format=self.memory_format, non_blocking=self.is_pin_memory), targets.to(self.device,
                                                non_blocking=self.is_pin_memory)

                with autocast(device_type=self.device.type, enabled=self.use_amp) :
                    output = ref_model(inputs)

                    if hasattr(output, "logits") :
                        output = output.logits

                y_true.append(targets.detach().cpu())

                prob = self._decode(output, threshold=probs_threshold, return_probs=True)
                pred = self._decode(output, threshold=probs_threshold, return_probs=False)

                y_prob.append(prob.detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
        y_pred = torch.cat(y_pred)

        if "clf" in self.mode :
            labels = torch.unique(y_true).cpu().tolist()
            metric_params = {k: v for k, v in self.metric_info.items() if k != "metric"}

            if self.mode == "clf_binary" :
                y_true = y_true.view(-1)
                y_prob = y_prob.view(-1)
                y_pred = y_pred.view(-1)

            acc = Accuracy(**metric_params)(y_pred, y_true)
            precision = Precision(**metric_params)(y_pred, y_true)
            recall = Recall(**metric_params)(y_pred, y_true)
            f1_score = F1Score(**metric_params)(y_pred, y_true)
            auc = AUROC(**metric_params)(y_prob, y_true.long())
            pr_auc = AveragePrecision(**metric_params)(y_prob, y_true.long())

            print(f"\nEvaluation results : \n\tacc : {acc:.4f} \n\tprecision : {precision:.4f} \n\trecall : {recall:.4f} \n\tf1-score : {f1_score:.4f} \n\tAUC : {auc:.4f}\n\tPR-AUC : {pr_auc:.4f}")

            if show_conf_matrix :
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                display_conf_matrix(cm)

        else :
            mae = MeanAbsoluteError()(y_true, y_pred)
            rmse = MeanSquaredError()(y_true, y_pred)

            print(f"\nEvaluation results ~ RMSE : {rmse:.4f} - MAE : {mae:.4f}\n")


    @final
    def show_history(self, show_lr: bool = False, save_fig: bool = False, save_root: str = "./runs/") :
        self.history.show(metric=self.metric, show_lr=show_lr, save_fig=save_fig, save_root=save_root)


    @final
    def show_info(self, input_size) :
        print("\nParameters : \n", summary(self.model, input_size=input_size))


    @classmethod
    def load_model(cls, model: nn.Module, model_path: str | None = None, criterion = None, dataloader_params: dict | None = None,
                   metric_info: dict | None = None, device = torch.device("cpu")) :

        if model_path is None :
            raise ValueError(f"You need to provide the path of the model !")
        else :
            if not Path(model_path).exists() :
                raise FileNotFoundError(f"No model found at {model_path} !")

        ckpt = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))  # weigths_only=True

        model.load_state_dict(ckpt["model"])
        model.to(device)

        batch_size = ckpt.get("batch_size", 64)

        ckpt_stuff = {
            "last_epoch": ckpt.get("last_epoch", 0),
            "lr": ckpt.get("lr", 1e-3),
            "optimizer": ckpt.get("optimizer", None), # optimizer.load_state_dict(ckpt_stuff["optimizer"])
            "scheduler": ckpt.get("scheduler", None), # callbacks.scheduler.load_state_dict(ckpt_stuff["scheduler"])
        }

        print("\nModel has been loaded with success !\n")

        return cls(
            model=model,
            batch_size=batch_size,
            criterion=criterion,
            metric_info=metric_info,
            dataloader_params=dataloader_params,
            device=device,
        ), ckpt_stuff

    @final
    def load_history(self, history_path: str | None = None) :
        if history_path is None :
            raise ValueError(f"You need to provide the path of the history !")
        else :
            if not Path(history_path).exists() :
                raise FileNotFoundError(f"No history found at {history_path} !")

        history = torch.load(history_path, weights_only=True)

        # n_epochs = history.get("n_epochs", 0)
        train_scores = history.get("train_scores", [])
        train_losses = history.get("train_losses", [])
        valid_scores = history.get("valid_scores", [])
        valid_losses = history.get("valid_losses", [])
        lr_list = history.get("lr_list", [])

        history_data = dict(
            train_scores=train_scores,
            train_losses=train_losses,
            valid_scores=valid_scores,
            valid_losses=valid_losses,
            lr_list=lr_list,
        )

        self.history = History(mode=self.mode)
        self.history.rebuild(**history_data)

        print("History has been loaded with success !\n")

    @final
    def load_ema(self, model_path: str | None = None) :
        if self.model is not None :
            if model_path is None :
                raise ValueError(f"You need to provide the path of the model !")
            else :
                if not Path(model_path).exists() :
                    raise FileNotFoundError(f"No history found at {model_path} !")

            ckpt = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))

            if "model_ema" in ckpt :
                self.model_ema = ModelEmaV2(self.model, device=self.device)
                self.model_ema.module.load_state_dict(ckpt["model_ema"])

                print("EMA model has been loaded with success !\n")


    def to_onnx(self, input_dim, filename: str = "model") :
        if self.model is not None :
            dummy_input = torch.randn(1, input_dim)

            self.model.eval()

            torch.onnx.export(
                self.model,
                dummy_input,
                f"./{filename}.onnx",
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )

            print("Model has been exported to onnx with success !\n")


