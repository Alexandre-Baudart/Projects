from pathlib import Path

import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import Literal

from sklearn.model_selection import train_test_split
from data.preprocessing import ProjectPreprocessor

def to_numpy(data, dtype=np.float32):
    if hasattr(data, "to_numpy"):
        return data.to_numpy(dtype=dtype)

    return np.asarray(data, dtype=dtype)

def split_df(df: DataFrame, train_size: float = 0.8, target: str | None = None) :
    return train_test_split(df, train_size=train_size, random_state=42, stratify=df[target])

def split_set(X: DataFrame, y, train_size: float = 0.80) :
    return train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

class Dataset_ :
    def __init__(self,
                 data_path: str | None = None,
                 train_path: str | None = None,
                 calib_path: str | None = None,
                 test_path: str | None = None,
                 dropped_cols: list | None = None,
                 binarization_info: dict | None = None,
                 show_df_info: bool = False) :

        self.data_path = data_path
        self.train_path = train_path
        self.calib_path = calib_path
        self.test_path = test_path

        self.dropped_cols = dropped_cols
        self.bin_info = binarization_info
        self.show_df_info = show_df_info

        self.data_df = None
        self.train_df = None
        self.calib_df = None
        self.test_df = None

        self.preprocessor = None
        self.preprocess = None

    def split_data_csv(self, target: str | None = None, train_size: float = 0.8, calib_size: float = 0.0):
        if self.data_path is not None:
            data_path = Path(self.data_path)
            root = data_path.parent

            filename = data_path.stem

            train_filename = filename + "_train.csv"
            train_path = root / train_filename
            test_filename = filename + "_test.csv"
            test_path = root / test_filename

            if train_path.exists() and test_path.exists():
                return
            else:
                data_df = self._load_csv(data_path, no_drop_cols=True)
                data_df = self._binarize_target(data_df)

                train_df, test_df = split_df(data_df, train_size=train_size, target=target)

                if calib_size > 0.0:
                    calib_filename = filename + "_calib.csv"
                    calib_path = root / calib_filename

                    train_df, calib_df = split_df(data_df, train_size=1-calib_size, target=target)

                    calib_df.to_csv(calib_path, index=False, encoding="utf-8")
                    
                train_df.to_csv(train_path, index=False, encoding="utf-8")
                test_df.to_csv(test_path, index=False, encoding="utf-8")

    def _load_csv(self, path: str | Path | None = None, no_drop_cols: bool = False) :
        df = None

        if path is not None:
            try:
                df = pd.read_csv(path)
                assert df is not None

                if not df.empty:
                    if self.show_df_info:
                        print("\nSome information about the dataset :\n")
                        print(df.info())  # types des variables
                        print(df.describe())  # statistiques descriptives
                        print(df.columns)  # noms exacts des columns

                if no_drop_cols and self.dropped_cols is not None:
                    df.drop(columns=self.dropped_cols, inplace=True)

            except FileNotFoundError:
                print(f"\"{path}\" not found !")

            except AssertionError:
                print(f"An error occurred during the loading of \"{path}\" !")

        return df

    def _binarize_target(self, df) :
        df_ = df

        if self.bin_info is not None:
            old_target = self.bin_info["old_target"]
            new_target = self.bin_info["new_target"]
            bin_threshold = self.bin_info["bin_threshold"]

            df_[new_target] = (df_[old_target] > bin_threshold).astype(int)
            df_.drop(columns=[old_target], inplace=True)

        return df_

    def get_original_df(self, binarize: bool = False):
        df = self._load_csv(self.data_path)

        if binarize:
            df = self._binarize_target(df)

        return df

    @staticmethod
    def _get_Xy(df: DataFrame, target: str | None = None):
        if df is None:
            raise RuntimeError("A DataFrame is required.")

        if target is None:
            raise ValueError("A target must be specified.")

        y = df[target]
        X = df.drop(columns=[target])

        return X, y

    def build_preprocessing(
            self,
            target: str | None = None,
            use_scaling: bool = False,
            cat_encoding: Literal["one_hot", "ordinal"] | None = "one_hot",
            encod_binary_features: bool = True,
    ) :
        if target is None:
            raise ValueError("A target must be specified.")

        self.preprocessor = ProjectPreprocessor(target=target)
        self.preprocess = self.preprocessor.build_preprocess(
            use_scaling,
            cat_encoding,
            encod_binary_features
        )

    def get_preprocess(self) :
        return self.preprocess

    def get_optim_set(self, target: str | None = None, search_set_size: float = 0.3):
        if self.train_df is None:
            self.train_df = self._load_csv(self.train_path)

        search_df, _ = split_df(self.train_df, train_size=search_set_size, target=target)

        return self._get_Xy(search_df, target=target)

    def get_train_set(self, target: str | None = None):
        if self.train_df is None:
            self.train_df = self._load_csv(self.train_path)

        return self._get_Xy(self.train_df, target=target)

    def get_calib_set(self, target: str | None = None):
        if self.calib_df is None:
            self.calib_df = self._load_csv(self.calib_path)

        return self._get_Xy(self.calib_df, target=target)

    def get_test_set(self, target: str | None = None):
        if self.test_df is None:
            self.test_df = self._load_csv(self.test_path)

        return self._get_Xy(self.test_df, target=target)

    def get_classes(self, target: str | None = None):
        data_df = self._load_csv(self.data_path)
        data_df = self._binarize_target(data_df)

        _, y = self._get_Xy(data_df, target=target)

        y = to_numpy(y).ravel()
        return np.unique(y)

    def to_npz(self, path : str | None = None, target: str | None = None) :
        if path is not None:
            X, y = self.get_test_set(target)

            X = X.to_numpy(dtype=np.float32)
            X = np.ascontiguousarray(X) # IMPORTANT

            y = np.asarray(y, dtype=np.int64)
            y = np.ascontiguousarray(y)

            # Contiguous check
            print("Contiguous X : ", X.flags["C_CONTIGUOUS"])
            print("Contiguous y : ", y.flags["C_CONTIGUOUS"])

            np.savez(f"{path}.npz", X=X, y=y)






