import torch
import numpy as np
from torch.utils.data import Dataset


def normalize(x) :
    return (x - x.mean()) / x.std()


class ProjectDataset(Dataset) :
    def __init__(self, X, y) :
        self.X = X
        self.y = y

        self.classes = np.unique(y)

    def _normalizeX(self) :
        return (self.X - self.X.mean()) / self.X.std()

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        X = self.X.iloc[idx].to_numpy(dtype=np.float32)
        y = self.y.iloc[idx]

        X = torch.from_numpy(X)
        X = normalize(X)

        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return X, y

    def get_labels(self):
        return torch.as_tensor(self.y.to_numpy(), dtype=torch.float)

