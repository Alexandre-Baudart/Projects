import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
import os
from typing import Literal


def build_table(cols: list, rows: list, save_path: str = None) :
    fig, ax = plt.subplots()
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=cols, loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    if save_path is not None :
        plt.savefig(save_path)

    plt.show()


def display_conf_matrix(cm, mode: Literal["clf_binary", "clf_multiclass", "clf_multilabel"] = "clf_binary", class_names: list = None, save_cm: bool = False, save_root: str = "./") :
    if mode in ["clf_binary", "clf_multiclass"] :
        df_cm = DataFrame(cm, index=class_names, columns=class_names)
        fig, ax = plt.subplots(figsize=(12, 5))

        print(cm)

        sns.heatmap(df_cm, annot=True, fmt="d", linewidths=0.5, ax=ax)
        plt.xlabel("Prediction")
        plt.ylabel("True")

    else :
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
        axes = axes.flatten()

        for i, (matrix, ax) in enumerate(zip(cm, axes)) :
            sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap="Blues", cbar=False)

            ax.set_title(class_names[i], fontweight="bold")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("True")

    plt.tight_layout()

    if save_cm :
        os.makedirs(save_root, exist_ok=True)
        plt.savefig(os.path.join(save_root, "conf_matrix.png"))

    plt.show()


def check_missing_values(df, col: str = None) :
    df.replace(["NA", "null", ""], pd.NA)

    if col is not None :
        if col not in df.columns :
            raise ValueError(f"{col} n'est pas une feature du dataset !")
        else : df[col].isna().sum()

    missing_values = df.isna().sum().sort_values(ascending=False)

    print("\nValeurs manquantes par feature : ", missing_values, sep="\n\n")


def heatmap_correlation(df, dropped_cols: list = None) :
    if dropped_cols is not None : df = df.drop(columns=dropped_cols)

    plt.figure(figsize=(12,8))
    corr = df.corr(numeric_only=True).round(2)

    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.tight_layout()
    plt.show()


def check_count(df, target: str, axes_names : tuple = None) :
    plt.figure()
    sns.countplot(df[target])

    if axes_names is not None :
        plt.xlabel(axes_names[0])
        plt.ylabel(axes_names[1])

    plt.tight_layout()
    plt.show()








