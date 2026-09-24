import numpy as np
import pandas as pd
from typing import Literal

from data.preprocessing import ProjectPreprocessor
from .utils import split_set

class Dataset_ :
    def __init__(self, train_size: float = 0.8) :
        self.df = None

        self.train_size = train_size

        self.preprocessor = None
        self.preprocess = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_csv(self, data_path: str | None = None, show_info: bool = False, dropped_cols: list | None = None) -> None :
        df = None

        if data_path is not None:
            try:
                df = pd.read_csv(data_path)
                assert df is not None

                if not df.empty:
                    if show_info:
                        print("\nSome information about the dataset :\n")
                        print(df.info())  # types des variables
                        print(df.describe())  # statistiques descriptives
                        print(df.columns)  # noms exacts des columns

                if dropped_cols is not None:
                    df.drop(columns=dropped_cols, inplace=True)

            except FileNotFoundError:
                print(f"\"{data_path}\" not found !")

            except AssertionError:
                print(f"An error occurred during the loading of \"{data_path}\" !")

        self.df = df

    def binarize_target(self, old_target, new_target, bin_threshold: int = 0) :
        self.df[new_target] = (self.df[old_target] > bin_threshold).astype(int)
        self.df.drop(columns=[old_target], inplace=True)

    def _split_data(self, target: str, train_size: float = 0.8):
        df = self.df.copy()

        y = df[target]
        X = df.drop(columns=[target])

        return split_set(X, y, train_size=train_size)

    def apply_preprocessing(
            self,
            target: str,
            use_scaling: bool = False,
            cat_encoding: Literal["one_hot", "ordinal"] | None = "one_hot",
            encod_binary_features: bool = True,
    ) :
        self.preprocessor = ProjectPreprocessor(target=target)
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(target=target, train_size=self.train_size)

        self.preprocess = self.preprocessor.build_preprocess(
            use_scaling,
            cat_encoding,
            encod_binary_features
        )

    def get_preprocess(self) :
        return self.preprocess

    def get_optim_set(self, search_set_size: float = 0.3):
        X_search, _, y_search, _ = split_set(self.X_train, self.y_train, train_size=search_set_size)
        return X_search, y_search

    def get_train_set(self):
        return self.X_train, self.y_train

    def get_test_set(self):
        return self.X_test, self.y_test

    def to_npz(self, path : str | None = None) :
        if path is not None :
            X = self.X_test.to_numpy(dtype=np.float32)
            y = np.asarray(self.y_test, dtype=np.int64)

            np.savez(f"{path}.npz", X=X, y=y)