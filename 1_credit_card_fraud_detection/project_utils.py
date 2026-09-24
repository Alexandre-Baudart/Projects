import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

def load_csv(file_path: str | None = None, show_info: bool = False) :
    df = None

    if file_path is not None :
        try :
            df = pd.read_csv(file_path)
            assert df is not None

            if not df.empty :
                if show_info :
                    print("\nSome information about the dataset :\n")
                    print(df.info()) # types des variables
                    print(df.describe()) # statistiques descriptives
                    print(df.columns) # noms exacts des columns

        except FileNotFoundError:
            print(f"\"{file_path}\" not found !")
            return None

        except AssertionError:
            print(f"An error occurred during the loading of \"{file_path}\" !")
            return None

    return df


def split_data(df, target : str | None = None, dropped_cols: list | None = None, used_cols: list | None = None, train_size = 0.8) :
    if dropped_cols is not None : df = df.drop(columns=dropped_cols)
    if used_cols is not None :
        used_cols = used_cols.copy()
        if target is not None : used_cols.append(target)
        df = df[used_cols]

    y = df[target] if target is not None else None
    X = df.drop(columns=target) if target is not None else df

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=train_size)

    num_cols = X_train.select_dtypes(include=["number"]).columns
    train_median = X_train[num_cols].median()

    X_train[num_cols] = X_train[num_cols].fillna(train_median)
    X_test[num_cols] = X_test[num_cols].fillna(train_median)

    X_train = X_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.int32)
    X_test = X_test.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.int32)

    return X_train, X_test, y_train, y_test


def split_set(X, y, train_size: float = 0.67) -> list :
    return train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)


def compute_pos_weight(dataset, eps: float = 1e-6) :
    y = dataset.get_labels()

    pos = y.sum(dim=0)
    neg = y.shape[0] - pos

    pos_weight = torch.log1p(neg / (pos + eps))

    return pos_weight


def to_npz(X, y, filename="dataset"):
    X = X.to_numpy(dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.savez(f"{filename}.npz", X=X, y=y)


def plot_roc_curve(y_true, y_pred) :
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_pr_auc_curve(y_true, y_pred):
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)

    pr_auc = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.plot(precision, recall, label=f"ROC AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()