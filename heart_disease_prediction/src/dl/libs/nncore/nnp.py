import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import final, Literal

import torch
from torchinfo import summary
import torch.nn as nn

from torch.amp import GradScaler, autocast
from timm.utils import ModelEmaV2
from sklearn.metrics import confusion_matrix, log_loss

from rich.progress import track
# from tqdm import tqdm

from ..history import History
from ..utils import (
    criterion_config,
    train_valid_loaders,
    metric_config,
    to_loader,
    brier_score,
    ECE
)
from ..callbacks import TimeMeasuring, TrainResultsMonitoring, BestModelCallback, CheckpointCallback, \
    ProgressiveUnfreezingCallback

class NNP:
    """
        Neural Network Pipeline
    """

    def __init__(self, model = None,
                 mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel", "reg"] = "clf_binary",
                 batch_size: int = 64, criterion = None, precision: int | str = 32, memory_format=None,
                 metric_info: dict | None = None,
                 dataloader_params: dict | None = None, device: torch.device = torch.device("cpu")):

        self.model = model
        self.mode = mode

        self.last_epoch = 0

        self.lr = None
        self.optimizer = None
        self.batch_size = batch_size

        self.model_ema = None

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = criterion_config(mode)

        if metric_info is None:
            self.metric_info = {
                "metric": "acc",
                "task": "binary"
            }
        else:
            self.metric_info = metric_info

        self.score_fn, self.metric = metric_config(mode, metric_info)

        self.precision = precision

        if self.precision == 32:
            self.use_amp = False
        elif self.precision in [16, "16-mixed"]:
            self.use_amp = (device.type == "cuda")
        else:
            raise ValueError("Unsupported precision")

        self.memory_format = memory_format

        self.use_gpu = (device != torch.device("cpu"))
        self.device = device

        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        self.dataloader_params = dataloader_params
        self.is_pin_memory = self.dataloader_params.get("pin_memory",
                                                        False) if self.dataloader_params is not None else False

        self.batch_proc_fn = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.callbacks = None
        self.history = History(mode=mode)

    def _batch_processing(self, inputs, targets):
        if self.batch_proc_fn is not None:
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
    def _compute_score(self, y_true, y_pred):
        self.score_fn.reset()

        y_true = y_true.long()

        self.score_fn.update(y_pred, y_true)
        score = self.score_fn.compute()

        return score

    def train(self, dataset=None, n_epochs: int = 10, lr: float = 1e-3, optimizer = None,
              accumulate_grad_batches: int = 1,
              clip_grad_norm: bool = False, batch_proc_fn = None,
              use_ema: bool = False, ema_decay: float = 0.999,
              unfreezing_schedule = None, callbacks: dict | None = None,
              save_root: str | None = None,
              save_best_model: bool = False, enable_checkpoints: bool = False):

        if self.model is None:
           raise ValueError("A model must be provided!")

        if dataset is None:
            raise ValueError("A train dataset must be provided!")

        # preprocess = dataset.get_preprocess() if hasattr(dataset, "get_preprocess") else None
        self.lr = lr
        self.batch_proc_fn = batch_proc_fn

        if optimizer is None:
            raise ValueError("An optimizer must be provided!")
        else:
            self.optimizer = optimizer

        if use_ema and self.model_ema is None:
            self.model_ema = ModelEmaV2(self.model, decay=ema_decay, device=self.device)

        self.callbacks = {} if callbacks is None else callbacks
        self.callbacks["time_measuring"] = TimeMeasuring()
        self.callbacks["tr_monitoring"] = TrainResultsMonitoring()

        if unfreezing_schedule is not None:
            self.callbacks["progressive_unfreezing"] = ProgressiveUnfreezingCallback(self.model,
                                                                                     unfreezing_schedule)

        if self.train_loader is None and self.valid_loader is None:
            self.train_loader, self.valid_loader = train_valid_loaders(dataset, batch_size=self.batch_size,
                                                                       dataloader_params=self.dataloader_params)

        if save_best_model:
            self.callbacks["best_model"] = BestModelCallback(
                self.model,
                self.optimizer,
                self.batch_size,
                self.callbacks["scheduler"],
                self.history,
                self.model_ema,
                save_root
            )

        if enable_checkpoints:
            self.callbacks["checkpoint"] = CheckpointCallback(
                self.model,
                # preprocess,
                self.optimizer,
                self.batch_size,
                self.callbacks["scheduler"],
                self.history,
                self.model_ema,
                save_root
            )

        for cb in self.callbacks.values():
            cb.on_train_begin()

        print("\n=== Training ===")

        for epoch in range(self.last_epoch, n_epochs):
            try:
                for cb in self.callbacks.values():
                    cb.on_epoch_begin(epoch)

                self.model.train()
                self.optimizer.zero_grad()

                print(f"\nEpoch {epoch + 1}/{n_epochs} : \n")

                with torch.enable_grad():
                    for batch, (inputs, targets) in enumerate(track(self.train_loader, description="Training...")):
                        if self.use_gpu:
                            inputs, targets = inputs.to(self.device, memory_format=self.memory_format,
                                                        non_blocking=self.is_pin_memory), targets.to(
                                self.device, non_blocking=self.is_pin_memory)

                        with autocast(device_type=self.device.type, enabled=self.use_amp):
                            inputs, targets = self._batch_processing(inputs, targets)

                            output = self.model(inputs)

                            if hasattr(output, "logits"):
                                output = output.logits

                            # targets = self._targets_processing(targets)

                            loss = self.criterion(output, targets)

                        loss = loss / accumulate_grad_batches

                        if self.use_amp:
                            self.scaler.scale(loss).backward()

                            if (batch + 1) % accumulate_grad_batches == 0:
                                if clip_grad_norm:
                                    self.scaler.unscale_(self.optimizer)
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                        else:
                            loss.backward()

                            if (batch + 1) % accumulate_grad_batches == 0:
                                if clip_grad_norm:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                                self.optimizer.step()
                                self.optimizer.zero_grad()

                        if use_ema:
                            self.model_ema.update(self.model)

                        for cb in self.callbacks.values():
                            cb.on_batch_end(batch, logs={"loss": loss.item()})

                y_true_train, y_pred_train, train_loss = self._validate(used_loader="train",
                                                                        track_description="Validating (training set)...")
                y_true_valid, y_pred_valid, valid_loss = self._validate(used_loader="validation",
                                                                        track_description="Validating (validation set)...")

                train_score = self._compute_score(y_true_train, y_pred_train)
                valid_score = self._compute_score(y_true_valid, y_pred_valid)

                lr = self.callbacks["scheduler"].current_lr if "scheduler" in self.callbacks else self.lr

                for cb in self.callbacks.values():
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
                print(f"\nQuick stop at epoch {epoch + 1}.")
                break

        results = {
            "duration_seconds": -1.0,
            "n_train_samples": len(dataset),
            "best_train_score": -1.0,
            "best_valid_score": -1.0,
        }

        for name, cb in self.callbacks.items():
            if name == "tr_monitoring":
                results["best_train_score"], results["best_valid_score"] = cb.on_train_end()
            elif name == "time_measuring":
                results["duration_seconds"] = cb.on_train_end()
            else:
                cb.on_train_end()

        return results

    @final
    def _decode(self, logits, threshold: float = 0.5, return_probs: bool = False):
        if self.mode == "clf_multiclass":
            return logits # torch.argmax(logits, dim=1)

        elif self.mode == "clf_binary" or self.mode == "clf_multilabel":
            probs = torch.sigmoid(logits)

            if return_probs:
                return probs

            return (probs > threshold).float()

        elif self.mode == "reg":
            return logits.squeeze(-1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _validate(self, used_loader: str = "validation", track_description: str = "Validating..."):
        y_true = []
        y_pred = []
        valid_loss = []

        ref_model = self.model if self.model_ema is None else self.model_ema.module

        ref_model.eval()

        if used_loader == "train":
            loader = self.train_loader
        elif used_loader == "test":
            loader = self.test_loader
        else:
            loader = self.valid_loader

        return_probs = True if self.metric in ["AUC", "PR-AUC"] else False

        with torch.no_grad():
            for inputs, targets in track(loader, description=track_description):
                if self.use_gpu:
                    inputs, targets = inputs.to(self.device, memory_format=self.memory_format,
                                                non_blocking=self.is_pin_memory), targets.to(self.device,
                                                                                             non_blocking=self.is_pin_memory)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = ref_model(inputs)

                    if hasattr(output, "logits"):
                        output = output.logits

                    # targets = self._targets_processing(targets)
                    loss = self.criterion(output, targets)
                    valid_loss.append(loss.item())

                y_true.append(targets.detach().cpu())

                pred = self._decode(output, return_probs=return_probs)
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        avg_valid_loss = sum(valid_loss) / len(valid_loss)

        return y_true, y_pred, avg_valid_loss

    def test(self, dataset_test, track_description: str = "Testing..."):
        if self.dataloader_params is not None:
            self.test_loader = to_loader(dataset_test, batch_size=self.batch_size, **self.dataloader_params)
        else:
            self.test_loader = to_loader(dataset_test, batch_size=self.batch_size)

        print("\n=== Test ===\n")

        y_true_test, y_pred_test, test_loss = self._validate(used_loader="test", track_description=track_description)

        test_score = self._compute_score(y_true_test, y_pred_test)
        print(f"Test results ~ Test {self.metric} : {test_score:.4f} - Test loss : {test_loss:.4f}\n")

        return y_true_test, y_pred_test

    def predict(self, dataset, return_probs: bool = False, track_description: str = "Inferencing..."):
        y_pred = []
        y_prob = []

        ref_model = self.model if self.model_ema is None else self.model_ema.module
        ref_model.eval()

        if self.dataloader_params is not None:
            loader = to_loader(dataset, batch_size=self.batch_size, **self.dataloader_params)
        else:
            loader = to_loader(dataset, batch_size=self.batch_size)

        if track_description == "Inferencing...":
            print("\n=== Inference ===\n")

        with torch.inference_mode():
            for batch in track(loader, description=track_description):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                if self.use_gpu:
                    inputs = inputs.to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = ref_model(inputs)

                pred = self._decode(output)
                y_pred.append(pred.detach().cpu())

                if return_probs:
                    prob = self._decode(output, return_probs=True)
                    y_prob.append(prob.detach().cpu())

        y_pred = torch.cat(y_pred)

        if return_probs:
            y_prob = torch.cat(y_prob)
            return y_pred, y_prob
        else:
            return y_pred

    def _display_conf_matrix(self, y_true, y_pred, labels, class_names: list | None = None, save_root: str | None = None):
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
    def evaluate(self,
                 dataset,
                 metrics: list | None = None,
                 calib_eval: bool = False,
                 conf_matrix: bool = False,
                 save_root: str | None = None,
                 track_description: str = "Evaluating..."):

        if metrics is None:
            if self.mode in ["clf_binary", "clf_multiclass", "clf_multilabel"]:
                metrics = ["acc", "precision", "recall", "f1-score", "auc", "pr_auc"]
            else:
                metrics = ["mae", "rmse"]

        results = {}

        print("\n=== Evaluation ===\n")

        y_true = dataset.get_labels(dtype=torch.int32)
        y_pred, y_prob = self.predict(dataset, return_probs=True, track_description=track_description)

        if "clf" in self.mode:
            labels = torch.unique(y_true).cpu().tolist()
            metric_info = {k: v for k, v in self.metric_info.items() if k != "metric"}

            if self.mode == "clf_binary" :
                y_true = y_true.view(-1)
                y_prob = y_prob.view(-1)
                y_pred = y_pred.view(-1)

            for metric in metrics:
                metric_info["metric"] = metric
                score_fn, metric_name = metric_config(mode=self.mode, metric_info=metric_info)

                if metric == "auc" or metric == "pr_auc":
                    results[metric_name] = float(score_fn(y_prob, y_true))
                else:
                    results[metric_name] = float(score_fn(y_pred, y_true))

            print(f"\nEvaluation results :")
            for metric, score in results.items():
                print(f"\t{metric} : {score:.4f}")

            if calib_eval :
                results["NLL"] = log_loss(y_true, y_prob)
                results["Brier-Score"] = brier_score(y_true, y_prob, self.mode)
                results["ECE"] = ECE(y_true, y_prob)

                print(f"\nNLL : {results["NLL"]:.4f}\nBrier-Score : {results["Brier-Score"]:.4f}\nECE : {results["ECE"]:.4f}")

            if conf_matrix:
                self._display_conf_matrix(
                    y_true=y_true,
                    y_pred=y_pred,
                    labels=labels,
                    save_root=save_root
                )

        else:
            print(f"\nEvaluation results :")
            for metric in metrics:
                score_fn, metric_name = metric_config(mode=self.mode, metric_info=dict(metric=metric))

                results[metric_name] = float(score_fn(y_true, y_pred))

                print(f"\t{metric_name} : {results[metric_name]:.4f}\n")

        return results

    def check_high_confidence_bias(self, dataset, threshold= 0.9) -> tuple :
        _, y_prob = self.predict(dataset, return_probs=True)
        y_true = dataset.get_labels()

        # Isolation of the highly confident predictions
        mask = y_prob >= threshold
        n_high_conf = torch.sum(mask)

        if n_high_conf == 0:  # no prediction over the threshold
            return False, None

        # Calculation of the proportion of these prediction in the dataset
        # pct = n_high_conf / len(y_prob)

        # Comparison between the mean confidence and the real precision.
        avg_confidence = torch.mean(y_prob[mask])
        actual_acc = torch.mean(y_true[mask])

        # Overconfidence gap
        gap = avg_confidence - actual_acc

        # Alert threshold : if the gap is over 5% -> a recalibration is required
        if gap > 0.05:
            return True, gap
        else:
            return False, gap

    def benchmark(self, dataset, n_iterations: int = 100):
        import time
        import numpy as np

        print("\n=== Benchmark ===\n")

        # Warm-up
        for _ in range(10):
            self.predict(dataset, track_description="Benchmarking (warm-up)...")

        latencies = []

        # n_repeats to reduce the impact of perf_counter() cost and
        # the "system noise" (important for models with high inference speed)

        # n_repeats = 10

        print("\n=== Benchmark ===\n")

        for _ in range(n_iterations):
            start = time.perf_counter()

            self.predict(dataset, track_description="Benchmarking...")

            end = time.perf_counter()

            latencies.append(end - start)

        latencies = np.array(latencies)

        mean_latency = latencies.mean()
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        throughput = float(len(dataset) / mean_latency)

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

    @final
    def show_history(self, show_lr: bool = False, save_fig: bool = False, save_root: str = "./runs/"):
        self.history.show(metric=self.metric, show_lr=show_lr, save_fig=save_fig, save_root=save_root)

    @final
    def show_info(self, input_size):
        if self.model is not None:
            print("\nParameters : \n", summary(self.model, input_size=input_size))

    @classmethod
    def load_model(cls, model: nn.Module, model_path: str | None = None,
                   criterion = None, dataloader_params: dict | None = None,
                   metric_info: dict | None = None, device=torch.device("cpu")):

        if model_path is None:
            raise ValueError(f"You need to provide the path of the model !")
        else:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"No model found at {model_path} !")

        ckpt = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))  # weigths_only=True

        model.load_state_dict(ckpt["model"])
        model.to(device)

        batch_size = ckpt.get("batch_size", 64)

        """
        if preprocess_path is not None:
            preprocess = joblib.load(preprocess_path)
        else:
            preprocess = None
        """

        ckpt_stuff = {
            # "preprocess": preprocess,
            "last_epoch": ckpt.get("last_epoch", 0),
            "lr": ckpt.get("lr", None),
            "optimizer": ckpt.get("optimizer", None),  # optimizer.load_state_dict(ckpt_stuff["optimizer"])
            "scheduler": ckpt.get("scheduler", None),  # callbacks.scheduler.load_state_dict(ckpt_stuff["scheduler"])
        }

        print("\nNote: Model has been loaded with success!")

        return cls(
            model=model,
            batch_size=batch_size,
            criterion=criterion,
            metric_info=metric_info,
            dataloader_params=dataloader_params,
            device=device,
        ), ckpt_stuff

    @final
    def load_history(self, history_path: str | None = None):
        if history_path is None:
            raise ValueError(f"You need to provide the path of the history !")
        else:
            if not Path(history_path).exists():
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
    def load_ema(self, model_path: str | None = None):
        if self.model is not None:
            if model_path is None:
                raise ValueError(f"You need to provide the path of the model !")
            else:
                if not Path(model_path).exists():
                    raise FileNotFoundError(f"No history found at {model_path} !")

            ckpt = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))

            if "model_ema" in ckpt:
                self.model_ema = ModelEmaV2(self.model, device=self.device)
                self.model_ema.module.load_state_dict(ckpt["model_ema"])

                print("EMA model has been loaded with success !\n")

    def to_onnx(self, input_dim, filename: str = "model"):
        if self.model is not None:
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


