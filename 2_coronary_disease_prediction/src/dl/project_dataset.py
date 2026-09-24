import joblib
import os
from typing import Literal
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from data.dataset import Dataset_
from .libs.utils import split_set

def to_numpy(data, dtype=np.float32):
    if hasattr(data, "to_numpy"):
        return data.to_numpy(dtype=dtype)

    return np.asarray(data, dtype=dtype)

def to_tensor(data, dtype=np.float32):
    return torch.as_tensor(to_numpy(data, dtype=dtype))

def split_indices(labels, ratio: float = 0.2, use_stratified_split: bool = False):
    if use_stratified_split:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=ratio,
            random_state=42
        )
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=ratio,
            random_state=42
        )

    train_idx, valid_idx = next(splitter.split(np.zeros(len(labels)), labels))

    return train_idx, valid_idx

class ProjectTrainDataset:
    def __init__(self,
            dataset: Dataset_,
            target: str,
            preprocess_path: str | None = None,
            task: Literal["optim", "train", "calib"] = "train",
            valid_ratio: float = 0.2,
            use_stratified_split: bool = False) :

        self.dataset = dataset
        self.task = task
        self.classes = self.dataset.get_classes(target=target)

        self.preprocess_ = None
        self._load_preprocess_(preprocess_path)

        X_for_fit = None
        y_for_fit = None

        if task == "train":
            X, y = self.dataset.get_train_set(target=target)
            X_for_fit = X
            y_for_fit = y
        elif task == "optim":
            X, y = self.dataset.get_train_set(target=target)

            X_for_fit = X
            y_for_fit = y

            _, X, _, y = split_set(X, y, train_size=0.3)
        else:
            X, y = self.dataset.get_calib_set(target=target)

        train_idx, valid_idx = split_indices(
            labels=y,
            ratio=valid_ratio,
            use_stratified_split=use_stratified_split
        )

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        if self.preprocess_ is not None:
            if task == "train" or task == "optim":
                self.preprocess_.fit(X_for_fit, y_for_fit)

            X_train = self.preprocess_.transform(X_train)
            X_valid = self.preprocess_.transform(X_valid)

        self.X_train = to_tensor(X_train)
        self.y_train = to_tensor(y_train).unsqueeze(-1)
        self.X_valid = to_tensor(X_valid)
        self.y_valid = to_tensor(y_valid).unsqueeze(-1)

    def _load_preprocess_(self, path: str | None = None):
        if path is not None:
            process_path = os.path.join(path, "preprocess_.joblib")

            if os.path.exists(path):
                preprocess_ = joblib.load(process_path)
                print("\nNote: preprocess_ loaded with success!")

                self.preprocess_ = preprocess_

        self.preprocess_ = self.dataset.get_preprocess()

    def get_preprocess_(self) :
        return self.preprocess_

    def get_input_size(self):
        return self.X_train.shape[1]

    def get_n_classes(self):
        n_classes = len(self.classes)

        return n_classes if n_classes > 2 else 1

    def get_tab_transformer_stuff(self) :
        ct = self.preprocess_.named_steps["column_transformer"]

        num_cols = ct.transformers_[0][2]
        cat_cols = ct.transformers_[1][2]

        n_num = len(num_cols)
        n_cat = len(cat_cols)

        cat_idx = list(range(n_num, n_num + n_cat))

        cat_cardinalities = [
            len(c) + 1
            for c in ct.named_transformers_["cat"].categories_
        ]

        return n_num, n_cat, cat_idx, cat_cardinalities

    def shift_cat_indices(self, cat_idx):
        if self.preprocess_ is not None:
            self.X_train[:, cat_idx] += 1
            self.X_valid[:, cat_idx] += 1

    def __len__(self):
        return len(self.X_train)

class ProjectTestDataset(Dataset) :
    def __init__(self,
            dataset: Dataset_,
            target: str,
            preprocess_ = None):

        self.dataset = dataset

        self.preprocess_ = preprocess_

        X, y = dataset.get_test_set(target=target)

        if self.preprocess_ is not None:
            X_t = self.preprocess_.transform(X)
        else :
            X_t = X.copy()

        self.X = torch.asarray(to_numpy(X_t, dtype=np.float32))
        self.y = torch.asarray(to_numpy(y, dtype=np.float32)).unsqueeze(-1)

        self.classes = np.unique(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        return X, y

    def load_preprocess_(self, path: str | None = None):
        if path is not None:
            process_path = os.path.join(path, "preprocess_.joblib")

            if os.path.exists(path):
                preprocess_ = joblib.load(process_path)
                print("\nNote: preprocess_ loaded with success!")

                self.preprocess_ = preprocess_

        self.preprocess_ = self.dataset.get_preprocess()

    def set_preprocess_(self, preprocess_):
        self.preprocess_ = preprocess_

    def get_labels(self, dtype = torch.float32):
        return self.y.to(dtype)

    def get_input_size(self):
        return self.X.shape[1]

    def get_n_classes(self):
        return len(self.classes)

    def get_tab_transformer_stuff(self) :
        ct = self.preprocess_.named_steps["column_transformer"]

        num_cols = ct.transformers_[0][2]
        cat_cols = ct.transformers_[1][2]

        n_num = len(num_cols)
        n_cat = len(cat_cols)

        cat_idx = list(range(n_num, n_num + n_cat))

        cat_cardinalities = [
            len(c) + 1
            for c in ct.named_transformers_["cat"].categories_
        ]

        return n_num, n_cat, cat_idx, cat_cardinalities

    def shift_cat_indices(self, cat_idx):
        if self.preprocess_ is not None:
            self.X[:, cat_idx] += 1