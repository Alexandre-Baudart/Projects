import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Literal, final
from abc import ABC, abstractmethod

@abstractmethod
class ProjectDataset(ABC, Dataset):
    def __init__(self, X, y, **kwargs) :
        self.X = X
        self.y = y

        mean = kwargs.get("mean", None)
        std = kwargs.get("std", None)

        if mean and std:
            self._normalize(mean, std)

        self.classes = torch.unique(y)

    @final
    def _normalize(self, mean, std):
        return (self.X - mean) / std

    @final
    def get_labels(self):
        return self.y

    @final
    def __len__(self):
        return len(self.X)

    @final
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X, y

class ProjectTrainDataset(ProjectDataset):
    def __init__(self, X_train, y_train) :
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1)

        self.mean = X_train_t.mean()
        self.std = X_train_t.std()

        super().__init__(
            X_train_t,
            y_train_t,
            mean=self.mean,
            std=self.std
        )

    @final
    def get_mean_std(self):
        return self.mean, self.std

    @final
    def get_input_size(self):
        return self.X.shape[1]

    @final
    def get_n_classes(self):
        n_classes = len(self.classes)

        if n_classes == 2 :
            return 1
        else :
            return n_classes

class ProjectTestDataset(ProjectDataset):
    def __init__(self, X_test, y_test, **kwargs) :
        super().__init__(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test.astype(np.float32)).unsqueeze(-1),
            **kwargs
        )

