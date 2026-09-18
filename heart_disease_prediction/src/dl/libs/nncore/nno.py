import torch
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ..utils import (
    criterion_config,
    train_valid_loaders,
    optimizer_config,
    metric_config
)

def short_dirname(trial) :
    return f"trial_{trial.trial_id}"


def build_model(model, cfg) :
    return model(cfg)


def train_(config) :
    torch.set_num_threads(1)
    dataset = ray.get(config["data_ref"])

    search_space = config["search_space"]

    ban_list = []

    weight_decay = search_space.get("weight_decay", None)

    if weight_decay is None : weight_decay = 0
    else : ban_list.append("weight_decay")

    model_params = {k: v for k, v in search_space.items() if k not in ban_list}

    train_loader, valid_loader = train_valid_loaders(
        dataset,
        batch_size=config["batch_size"],
        dataloader_params=config["dataloader_params"]
    )

    model = ray.get(config["model_ref"])
    model = build_model(model, model_params).to(config["device"])

    optimizer_info = config["optimizer_info"]
    optimizer_info["weight_decay"] = weight_decay

    optimizer = optimizer_config(
        model_or_params=model,
        optimizer_info=optimizer_info
    )

    best_score = -1.0
    device = config["device"]

    for epoch in range(config["n_epochs"]) :
        model.train()

        with torch.enable_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                output = model(inputs)

                loss = config["criterion"](output, targets)

                loss.backward()
                optimizer.step()

        valid_val = validate_(model, valid_loader, config)

        if hasattr(valid_val, "item") :
            valid_val = valid_val.item()

        best_score = max(best_score, valid_val)

        tune.report(
            {
                config["metric"]: valid_val,
                "best_global_score": best_score,
            }
        )


def _decode(mode, logits, threshold: float = 0.5, return_probs: bool = False) :
    if mode == "clf_multiclass":
        return torch.argmax(logits, dim=1)

    elif mode == "clf_binary" or mode == "clf_multilabel":
        probs = torch.sigmoid(logits)

        if return_probs :
            return probs

        return (probs > threshold).float()

    elif mode == "reg":
        return logits.squeeze(-1)

    else:
        raise ValueError(f"Unknown mode: {mode}")
    

def compute_score_(score_fn, y_true, y_pred) :
    score_fn.reset()

    y_true = y_true.long()

    score_fn.update(y_pred, y_true)
    score = score_fn.compute()

    return score


def validate_(model, loader, config) :
    y_true = []
    y_pred = []

    device = config["device"]
    mode = config["nn_mode"]

    return_probs = True if config["metric"] in ["AUC", "PR-AUC"] else False

    model.eval()

    with torch.no_grad() :
        for inputs, targets in loader :
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)

            if hasattr(output, "logits") :
                output = output.logits

            pred = _decode(mode=mode, logits=output, return_probs=return_probs)

            y_true.append(targets.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    return compute_score_(config["score_fn"], y_true, y_pred)


class NNO :
    """
    Neural Network Optimizer
    """
    def __init__(self, model, dataset_train = None, nn_mode: str = "clf_binary", batch_size: int = 32, criterion = None, metric_info : dict | None = None, optim_mode: str = "max", n_epochs: int = 10,
                 dataloader_params: dict | None = None, save_best_model: bool = False, device = torch.device("cpu")) :

        self.model = model
        self.dataset_train = dataset_train

        self.nn_mode = nn_mode
        self.optim_mode = optim_mode

        if metric_info is None :
            metric_info = {
                "metric": "acc",
                "task": "binary"
            }

        self.score_fn, self.metric = metric_config(nn_mode, metric_info)

        self.search_space = None
        self.optimizer_info = None

        self.batch_size = batch_size
        self.criterion = criterion if criterion is not None else criterion_config(nn_mode)
        self.n_epochs = n_epochs

        self.dataloader_params = dataloader_params

        self.save_best_model = save_best_model
        self.device = device

        self.best_score = -1.0

    def optimize(self, search_space: dict | None = None, optimizer_info: dict | None = None, grace_period: int = 3, reduction_factor: int = 3,
                 n_samples: int = 5, max_concurrent_trials: int = 1, resources_per_trial: dict | None = None, verbose: int = 2) :

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        ray.init()

        self.search_space = search_space or {}
        self.optimizer_info = optimizer_info or {}

        scheduler = ASHAScheduler(
            metric=self.metric,
            mode=self.optim_mode,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )

        model_ref = ray.put(self.model)
        data_ref = ray.put(self.dataset_train)

        print(f"\n=== Optimization: {self.model.__class__.__name__}  ===\n")

        analysis = tune.run(
            train_,
            config={
                "search_space": {
                    **self.search_space
                },
                "model_ref": model_ref,
                "batch_size": self.batch_size,
                "data_ref": data_ref,
                "device": self.device,
                "criterion": self.criterion,
                "metric": self.metric,
                "score_fn": self.score_fn,
                "nn_mode": self.nn_mode,
                "optimizer_info": self.optimizer_info,
                "n_epochs": self.n_epochs,
                "dataloader_params": self.dataloader_params
            },
            num_samples=n_samples,
            scheduler=scheduler,
            resources_per_trial=resources_per_trial if resources_per_trial else {"cpu": 2},
            max_concurrent_trials=max_concurrent_trials,
            storage_path="C:/ray",
            trial_name_creator=short_dirname,
            verbose=verbose
        )

        best_config = analysis.get_best_config(metric=self.metric, mode=self.optim_mode)["search_space"]

        if best_config is not None:
            self.model = build_model(self.model, best_config)

            best_config = {k: v for k, v in best_config.items()
                           if k not in ["n_classes", "input_size", "n_num", "n_cat", "cat_idx", "cat_cardinalities"]}

            print(f"\nBest config :", best_config)

            if self.save_best_model :
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": best_config
                    },
                    "best_model.pt"
                )

        return best_config





