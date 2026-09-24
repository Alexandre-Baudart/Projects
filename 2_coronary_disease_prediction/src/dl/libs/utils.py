import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.optim as optim
import lion_pytorch as lion
import torch.nn as nn
import torch_directml

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score, AveragePrecision, Recall, Precision, F1Score, Metric

def random_init(seed:int = 42, device = torch.device("cpu")) :
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda" :
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id) :
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)

def to_loader(dataset, batch_size: int = 64, shuffle: bool | None = None, sampler = None, num_workers: int = 0, persistent_workers: bool = False,
              pin_memory: bool = False, prefetch_factor: int | None = None, collate_fn = None) :

    # Warning : keep num_workers = 0 for little datasets and batches quickly computed,
    # otherwise the multiprocessing cost (overhead) could become enormous compared to the actual work required.

    if num_workers > 0 :
        g = torch.Generator().manual_seed(42)
        g.manual_seed(42)
    else :
        g = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else None,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

def compute_sample_weights(dataset, eps=1e-6) :
    # extraction labels
    if isinstance(dataset, torch.utils.data.Subset):
        base_labels = dataset.dataset.get_labels()
        labels = base_labels[dataset.indices]
    else:
        labels = dataset.get_labels()

    # tensor -> numpy
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    labels = np.asarray(labels).astype(int).ravel()

    # class weight
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + eps)

    # sample weight
    sample_weights = class_weights[labels]

    return torch.tensor(sample_weights, dtype=torch.float32)

def train_valid_loaders(dataset, batch_size: int = 64, dataloader_params: dict | None = None) :
    dataloader_params = dataloader_params or {}

    def get_labels_from_any(dataset) :
        if hasattr(dataset, "labels") :
           return np.array(dataset.labels)

        elif hasattr(dataset, "y") :
            return np.array(dataset.y)

        elif hasattr(dataset, "get_labels") :
            return dataset.get_labels()

        elif isinstance(dataset, torch.utils.data.Subset) :
            parent_labels = get_labels_from_any(dataset.dataset)
            return parent_labels[dataset.indices]

        else :
            raise ValueError("The dataset provided does not have the attribute 'labels', 'y' or a 'get_labels' method and it is not a Subset either !")

    train_dataset = TensorDataset(
        dataset.X_train,
        dataset.y_train
    )

    valid_dataset = TensorDataset(
        dataset.X_valid,
        dataset.y_valid
    )

    use_weighted_sampler = dataloader_params.get("use_weighted_sampler", False)
    dataloader_params = {k: v for k, v in dataloader_params.items() if k != "use_weighted_sampler"}

    sampler = None

    if use_weighted_sampler :
        sample_weights = compute_sample_weights(train_dataset)

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    shuffle = True if sampler is None else None

    train_loader = to_loader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **dataloader_params)
    valid_loader = to_loader(valid_dataset, batch_size=batch_size, **dataloader_params)

    return train_loader, valid_loader

def load_dataset(dataset_name, batch_size: int = 64, need_loader: bool = False) :
    transform = ToTensor()

    dataset_train = None
    dataset_test = None
    train_loader = None
    valid_loader = None

    try :
        if dataset_name not in ("mnist", "cifar10") : raise ValueError(f"Erreur, le dataset \"{dataset_name}\" n'a pas été trouvé !\n")

        os.makedirs(os.path.dirname("../../../data"), exist_ok=True)

        if dataset_name == "mnist" :
            dataset_train = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
            dataset_test = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
        else :
            dataset_train = datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)
            dataset_test = datasets.CIFAR10(root="../data", train=False, transform=transform, download=True)

        if need_loader :
            train_loader, valid_loader = train_valid_loaders(dataset_train, batch_size=batch_size)

    except ValueError as e :
        print(e)

    if need_loader :
        return dataset_train, dataset_test, train_loader, valid_loader
    else :
        return dataset_train, dataset_test


def get_X(dataset) :
    # cas TensorDataset
    if isinstance(dataset, torch.utils.data.TensorDataset) :
        return dataset.tensors[0]

    # cas DataLoader
    if isinstance(dataset, DataLoader) :
        X_list = []
        for inputs, _ in dataset :
            X_list.append(inputs)
        return torch.cat(X_list, dim=0)

    # cas Tensor
    if isinstance(dataset, torch.Tensor) :
        return dataset

    raise TypeError(f"Impossible de récupérer les features pour le type {type(dataset)}")


def get_y(dataset) :
    # cas TensorDataset
    if hasattr(dataset, "tensors") :
        return dataset.tensors[1]

    # cas Subset (issu de random_split)
    elif hasattr(dataset, "indices") and hasattr(dataset, "dataset") :
        return get_y(dataset.dataset)[dataset.indices]

    # cas dataset venant de torchvision
    elif hasattr(dataset, "targets") :
        return torch.tensor(dataset.targets)

    raise TypeError(f"Impossible de récupérer les targets pour le type {type(dataset)}")


def select_device(type_: Literal["cpu", "gpu"] = "cpu") :
    if type_ == "cpu":
        return torch.device("cpu")
    else:
        if torch.cuda.is_available() : # pour NVIDIA
            print("\nGPU NVIDIA found and ready to be used with CUDA.\n\n----\n")
            return torch.device("cuda")
        elif torch_directml.is_available():  # pour AMD sur Windows
            print("\nGPU AMD found and ready to be used with DirectML.\n\n----")
            return torch_directml.device()
        else :
            return torch.device("cpu")

def optimizer_config(model_or_params = None, optimizer_info: dict | None = None) :
    OPTIMIZERS = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "lion": lion.Lion
    }

    if model_or_params is not None :
        optim_info = optimizer_info or {}

        optimizer_type = optim_info.get("type", "sgd")
        optim_params = {k: v for k, v in optim_info.items() if k != "type"}

        if optimizer_type in OPTIMIZERS :
            optimizer = OPTIMIZERS[optimizer_type]

            if isinstance(model_or_params, list) :
                return optimizer(model_or_params, **optim_params)
            else :
                return optimizer(filter(lambda p: p.requires_grad, model_or_params.parameters()), **optim_params)

        print("\nThe optimizer SGD will be used...\n")

        if isinstance(model_or_params, list) :
            return optim.SGD(model_or_params, **optim_params)
        else :
            return optim.SGD(model_or_params.parameters(), **optim_params)

    return None


class FocalLoss :
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) :
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets) :
        targets = targets.float()

        bce = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)

        loss = self.alpha * torch.pow(1 - pt, self.gamma) * bce
        return loss.mean()


def criterion_config(mode: str = "clf_binary", criterion_info: dict | None = None, device = torch.device("cpu")) :
    CRITERION = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce_logits": nn.BCEWithLogitsLoss,
        "multi_margin": nn.MultiMarginLoss,
        "focal_loss": FocalLoss,
        "nll": nn.NLLLoss,
        "mse": nn.MSELoss,
    }

    if criterion_info is not None :
        criterion_type = criterion_info.get("type", "")
        criterion_params = {k: v for k, v in criterion_info.items() if k != "type"}

        if criterion_type in CRITERION :
            if criterion_type == "bce_logits" and "pos_weight" in criterion_params :
                pos_weight = criterion_params["pos_weight"].to(device)
                criterion_params = {k: v for k, v in criterion_params.items() if k != "pos_weight"}

                return CRITERION["bce_logits"](pos_weight=pos_weight, **criterion_params)

            return CRITERION[criterion_type](**criterion_params)

    return nn.CrossEntropyLoss() if "clf" in mode else nn.MSELoss()


def apply_tta(model, inputs) :
    outputs = list()

    # prédiction normale
    outputs.append(model(inputs))

    # flip horizontal
    outputs.append(model(torch.flip(inputs, dims=[3])))

    # flip vertical
    # outputs.append(model(torch.flip(inputs, dims=[2])))

    # moyenne logits
    return torch.stack(outputs).mean(dim=0)

class BrierScore(Metric):
    def __init__(self, task: str = "binary", **kwargs):
        super().__init__(**kwargs)

        self.task = task

        self.add_state(
            "sum_squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )

        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
        )

    def update(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        if self.task == "binary":
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)

            error = (y_pred - y_true) ** 2

        else:
            error = (y_pred - y_true) ** 2

        self.sum_squared_error += error.sum()
        self.total += error.numel()

    def compute(self):
        return self.sum_squared_error / self.total

class ECE:
    def __init__(self, task: str = "binary", n_bins: int = 10, **kwargs):
        self.task = task
        self.n_bins = n_bins

    def __call__(self, y_prob, y_true):
        # Convertir en tenseurs de type float32 (obligé pour mean() notamment)
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        y_prob = torch.as_tensor(y_prob, dtype=torch.float32)

        # Retrouver la distribution des effectifs par bin
        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=y_prob.device)
        bin_assignments = torch.bucketize(y_prob, bin_edges)

        # bin_assignments reste entre 0 et n_bins - 1
        bin_assignments = torch.clamp(bin_assignments, 0, self.n_bins - 1)

        ece = torch.tensor(0.0)
        n_samples = len(y_true)

        # Calcul de la somme pondérée des écarts abs(vrai - prédit)
        for i in range(self.n_bins):
            # Sélection des éléments appartenant au bin i
            bin_mask = (bin_assignments == i)
            bin_size = torch.sum(bin_mask)  # item() -> renvoie l'élément hors du tenseur (ici: un int)

            if bin_size > 0:
                # Calculer la moyenne locale pour bin spécifique
                local_prob_true = torch.mean(y_true[bin_mask])
                local_prob_pred = torch.mean(y_prob[bin_mask])

                # Formule de l'ECE : (taille_bin / total) * |vrai - prédit|
                ece += (bin_size / n_samples) * torch.abs(local_prob_pred - local_prob_true)

        return ece.item()

class NLL:
    def __init__(self, task: str = "binary", **kwargs):
        self.task = task

    def __call__(self, y_prob, y_true):
        return log_loss(y_true, y_prob)

def metric_config(mode: str = "clf_binary", metric_info: dict | None = None) :
    CLF_METRICS = {
        "acc": {"metric": Accuracy, "name": "Accuracy"},
        "auc": {"metric": AUROC, "name": "ROC-AUC"},
        "pr_auc": {"metric": AveragePrecision, "name": "PR-AUC"},
        "recall": {"metric": Recall, "name": "Recall"},
        "precision": {"metric": Precision, "name": "Precision"},
        "f1_score": {"metric": F1Score, "name": "F1-Score"},
        "brier_score": {"metric": BrierScore, "name": "Brier-Score"},
        "ece": {"metric": ECE, "name": "ECE"},
        "nll": {"metric": NLL, "name": "NLL"},
    }
    REG_METRICS = {
        "mae": {"metric": MeanAbsoluteError, "name": "MAE"},
        "r2": {"metric": R2Score, "name": "R2"},
        "mse": {"metric": MeanSquaredError, "name": "MSE"}
    }

    if metric_info is not None :
        metric = metric_info.get("metric", "")

        if "clf" in mode :
            if metric is not None :
                task = metric_info.get("task", None)
                task = task or "binary"

                metric_params = {k: v for k, v in metric_info.items() if k != "task" and k != "metric"}

                if metric in CLF_METRICS :
                    res = CLF_METRICS[metric]
                    return res["metric"](task, **metric_params), res["name"]

        else :
            if metric in REG_METRICS:
                res = REG_METRICS[metric]
                return res["metric"](), res["name"]

    if mode == "clf_binary":
        return Accuracy("binary"), "Accuracy",
    elif mode == "clf_multiclass":
        return Accuracy("multiclass"), "Accuracy",
    elif mode == "clf_multilabel":
        return Accuracy("multilabel"), "Accuracy",
    else:
        return MeanSquaredError(), "MSE"

def load_model(model, model_path, device) :
    if model_path is None :
        raise ValueError(f"You need to provide the path of the model !")
    else:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No model found at {model_path} !")

    ckpt = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))

    model.load_state_dict(ckpt)
    model.to(device)

    return model

def split_set(X, y, train_size: float = 0.80) :
    return train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

