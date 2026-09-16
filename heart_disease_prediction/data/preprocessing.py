from typing import Literal
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
# pd.set_option("future.no_silent_downcasting", True)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_cols_without_flag: list | None = None,
        num_cols_with_flag: list | None = None,
        cat_cols: list | None = None,
        cat_ord_cols: list | None = None,
        cat_bin_cols: list | None = None,
    ):
        self.num_cols_without_flag = num_cols_without_flag
        self.num_cols_with_flag = num_cols_with_flag
        self.cat_cols = cat_cols
        self.cat_ord_cols = cat_ord_cols
        self.cat_bin_cols = cat_bin_cols

        self.imputation_values = {}

    def fit(self, X, y = None):
        X = X.copy()

        if self.num_cols_with_flag is not None:
            for col in self.num_cols_with_flag:
                self.imputation_values[col] = X[col].median()

        if self.num_cols_without_flag is not None:
            for col in self.num_cols_without_flag:
                self.imputation_values[col] = X[col].median()

        if self.cat_ord_cols is not None:
            for col in self.cat_ord_cols:
                self.imputation_values[col] = X[col].mode()[0]

        if self.cat_bin_cols is not None:
            for col in self.cat_bin_cols:
                self.imputation_values[col] = X[col].mode()[0]

        self.is_fitted_ = True # _ is a state marker in Scikit-learn == important for stateless transformers in order to notice a fitted state.
        # Here, "stateless" means that fit() has learned nothing from data (always applying the same process independently of the data provided).

        return self

    def transform(self, X):
        X = X.copy()
        self.feature_names_in_ = X.columns.to_numpy()

        # Note : flag peut-être inutile si ratio NaN < 5 % dataset.

        # === Variables numériques continues ===

        if self.num_cols_with_flag is not None :
            for col in self.num_cols_with_flag :
                # ajout d'un flag pour indiquer le manque de valeur (0 ou 1)
                X[f"{col}_missing"] = X[col].isnull().astype(int)

                # remplissage des valeurs manquantes par la médiane
                X[col] = pd.to_numeric(X[col].fillna(self.imputation_values[col]))

        if self.num_cols_without_flag is not None :
            for col in self.num_cols_without_flag :
                X[col] = pd.to_numeric(X[col].fillna(self.imputation_values[col]))

        # === Variables catégorielles ===

        if self.cat_cols is not None :
            for col in self.cat_cols :
                # remplissage des vals manquantes par "Unknown"
                X[col] = X[col].fillna("Unknown")

        # === Variables ordinales ===

        if self.cat_ord_cols is not None :
            for col in self.cat_ord_cols :
                X[f"{col}_missing"] = X[col].isnull().astype(int)

                # imputation par la valeur la plus fréquente
                X[col] = X[col].fillna(self.imputation_values[col])

        # === Variables catégoriques binaires ===

        # Note : pas de flag, car var simple, peu d'information structurelle, mode = suffisant.

        if self.cat_bin_cols is not None :
            for col in self.cat_bin_cols :
                if X[col].isna().any():
                    with pd.option_context("future.no_silent_downcasting", True):
                        X[col] = X[col].fillna(self.imputation_values[col])

        return X

    def get_feature_names_out(
            self,
            input_features=None
    ):

        if input_features is None:
            input_features = self.feature_names_in_

        feature_names = list(input_features)

        # Flags numériques
        if self.num_cols_with_flag is not None:
            feature_names.extend(
                [
                    f"{col}_missing"
                    for col in self.num_cols_with_flag
                ]
            )

        # Flags ordinaux
        if self.cat_ord_cols is not None:
            feature_names.extend(
                [
                    f"{col}_missing"
                    for col in self.cat_ord_cols
                ]
            )

        return np.array(feature_names, dtype=object)

class BinaryFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binary_mappings: dict | None = None):
        self.binary_mappings = binary_mappings

    def fit(self, X, y = None):
        if self.binary_mappings is None:
            return self

        missing_cols = set(self.binary_mappings) - set(X.columns)

        if missing_cols:
            raise ValueError(f"Missing columns from X : {missing_cols}")

        self.is_fitted_ = True

        return self

    def transform(self, X):
        X = X.copy()

        if self.binary_mappings is None:
            return X

        for col, mapping in self.binary_mappings.items():
            X[col] = X[col].map(mapping)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None

        return input_features

class ToStringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list | None = None):
        self.cols = cols

    def fit(self, X, y = None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        if self.cols is not None:
            X[self.cols] = X[self.cols].astype("str")

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None

        return input_features

class Preprocessor(ABC):
    def __init__(self, target: str | None = None):
        self.target = target
        self.pipeline = None

    @abstractmethod
    def build_preprocess(self):
        raise NotImplementedError("build_preprocess method is not implemented !")

    def fit(self, X, y = None):
        if self.pipeline is None:
            self.build_preprocess()

        self.pipeline.fit(X)

        return self

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Pipeline must be build before transform() !")

        return self.pipeline.transform(X)

    def fit_transform(self, X, y = None):
        if self.pipeline is None:
            self.build_preprocess()

        return self.pipeline.fit_transform(X, y)

class ProjectPreprocessor(Preprocessor):
    def __init__(self, target: str | None = None):
        super().__init__(target)

    def build_preprocess(
        self,
        use_scaling: bool = False,
        cat_encoding: Literal[
            "one_hot",
            "ordinal",

        ] | None = "one_hot",
        encod_binary_features: bool = True,
        encod_to_string: bool = True
    ):
        num_cols_with_flag = ["oldpeak", "trestbps", "thalch"]
        num_cols_without_flag = ["age", "chol"]
        cat_cols = ["thal", "restecg"]
        cat_ord_cols = ["ca", "cp", "slope"]
        cat_bin_cols = ["sex", "fbs", "exang"]

        binary_mappings = {
            "sex": {
                "Male": 1,
                "Female": 0
            },
            "fbs": {
                True: 1,
                False: 0
            },
            "exang": {
                True: 1,
                False: 0
            }
        }

        transformers = []

        missing_values = MissingValuesTransformer(
            num_cols_without_flag,
            num_cols_with_flag,
            cat_cols,
            cat_ord_cols,
            cat_bin_cols
        )

        transformers.append(
            ("missing_values", missing_values)
        )

        if encod_binary_features:
            transformers.append(
                (
                    "binary_features",
                    BinaryFeaturesTransformer(binary_mappings=binary_mappings)
                )
            )

        if encod_to_string:
            transformers.append(
                ("to_string", ToStringTransformer(cols=["ca"]))
            )

        num_cols = (num_cols_with_flag + num_cols_without_flag)
        num_cols += [f"{col}_missing" for col in num_cols_with_flag]
        num_cols += [f"{col}_missing" for col in cat_ord_cols]

        if encod_binary_features:
            num_cols += cat_bin_cols

        final_cat_cols = (cat_cols + cat_ord_cols)

        num_transformer = (StandardScaler() if use_scaling else "passthrough")

        if cat_encoding is None:
            self.pipeline = Pipeline(
                steps=transformers
            )
        else :
            if cat_encoding == "one_hot":
                cat_transformer = OneHotEncoder(
                    handle_unknown="ignore"
                )
            elif cat_encoding == "ordinal":
                cat_transformer = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            else:
                cat_transformer = "passthrough"

            column_transformer = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        num_transformer,
                        num_cols
                    ),
                    (
                        "cat",
                        cat_transformer,
                        final_cat_cols
                    )
                ],
                remainder="drop"
            )

            self.pipeline = Pipeline(
                steps=[
                    *transformers,
                    (
                        "column_transformer",
                        column_transformer
                    )
                ]
            )

        return self.pipeline

"""     
class Preprocessor(ABC) :
    def __init__(self, df, target : str | None = None) :
        self.df = df
        self.target = target
        self.preprocess = None

        self.imputation_values = {}

    def _fit_missing_values(
            self,
            X_train: pd.DataFrame,
            num_cols_without_flag: list | None = None,
            num_cols_with_flag: list | None = None,
            cat_ord_cols: list | None = None,
            cat_bin_cols: list | None = None,
    ):
        if num_cols_with_flag is not None:
            for col in num_cols_with_flag:
                self.imputation_values[col] = X_train[col].median()

        if num_cols_without_flag is not None:
            for col in num_cols_without_flag:
                self.imputation_values[col] = X_train[col].median()

        if cat_ord_cols is not None:
            for col in cat_ord_cols:
                self.imputation_values[col] = X_train[col].mode()[0]

        if cat_bin_cols is not None:
            for col in cat_bin_cols:
                self.imputation_values[col] = X_train[col].mode()[0]

    def _manage_missing_values(
            self,
            X,
            num_cols_without_flag : list | None = None, # ["chol"]
            num_cols_with_flag : list | None = None, # ["oldpeak", "trestbps", "thalch"]
            cat_cols : list | None = None, # ["thal", "restecg"]
            cat_ord_cols : list | None = None, # ["slope"],
            cat_bin_cols : list | None = None # ["sex", "fbs", "exang"]
    ) :
        X = X.copy()

        # Note : flag peut-être inutile si ratio NaN < 5 % dataset.

        # === Variables numériques continues ===

        if num_cols_with_flag is not None :
            for col in num_cols_with_flag :
                # ajout d'un flag pour indiquer le manque de valeur (0 ou 1)
                X[f"{col}_missing"] = X[col].isnull().astype(int)

                # remplissage des valeurs manquantes par la médiane
                X[col] = X[col].fillna(self.imputation_values[col]).infer_objects(copy=False)

        if num_cols_without_flag is not None :
            for col in num_cols_without_flag :
                X[col] = X[col].fillna(self.imputation_values[col]).infer_objects(copy=False)

        # === Variables catégorielles ===

        if cat_cols is not None :
            for col in cat_cols :
                # remplissage des vals manquantes par "Unknown"
                X[col] = X[col].fillna("Unknown").infer_objects(copy=False)

        # === Variables ordinales ===

        if cat_ord_cols is not None :
            for col in cat_ord_cols :
                X[f"{col}_missing"] = X[col].isnull().astype(int)

                # imputation par la valeur la plus fréquente
                X[col] = X[col].fillna(self.imputation_values[col]).infer_objects(copy=False)

        # === Variables catégoriques binaires ===

        # Note : pas de flag, car var simple, peu d'information structurelle, mode = suffisant.

        if cat_bin_cols is not None :
            for col in cat_bin_cols :
                X[col] = X[col].fillna(self.imputation_values[col]).infer_objects(copy=False)

        return X

    def _binarize_target(self) :
        df = self.df.copy()

        df["target_binary"] = (df["num"] > 0).astype(int)

        self.df = df

    def _encod_binary_features(self, binary_features: list | None = None) :
        df = self.df.copy()

        if binary_features is not None :
            for col in binary_features :
                values = df[col].dropna().unique()

                if len(values) != 2:
                    df[col] = df[col].map({values[0]: 1, values[1]: 0})

            self.df = df

    def _to_str(self, cols: list | None = None) :
        if cols is not None :
            df = self.df.copy()
            df[cols] = df[cols].astype("str")

            self.df = df

    def _split_data(self, dropped_cols: list | None = None, used_cols: list | None = None) :
        df = self.df.copy()

        if dropped_cols is not None: df = df.drop(columns=dropped_cols)
        if used_cols is not None:
            used_cols = used_cols.copy()
            if self.target is not None: used_cols.append(self.target)
            df = df[used_cols]

        y = df[self.target] if self.target is not None else None
        X = df.drop(columns=self.target) if self.target is not None else df

        return X, y

    def _build_column_transformer(self, num_cols, cat_cols, use_scaling: bool = False, cat_encoding: Literal["one_hot", "ordinal"] = "one_hot") :
        num_transformer = StandardScaler() if use_scaling else "passthrough"

        if cat_encoding == "one_hot" :
            cat_transformer = OneHotEncoder(handle_unknown="ignore")
        elif cat_encoding == "ordinal" :
            cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            cat_transformer = "passthrough"

        self.preprocess = ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
            remainder="drop")

    @abstractmethod
    def build_preprocess(self) :
        raise NotImplementedError("build_preprocess method is not implemented !")

class ProjectPreprocessor(Preprocessor) :
    def __init__(self, df, target) :
        super().__init__(df, target)

    def build_preprocess(
            self,
            use_scaling: bool = False,
            cat_encoding: Literal["one_hot", "ordinal"] = "one_hot",
            encod_binary_features: bool = True,
            train_size: float = 0.8
    ) :
        dropped_cols = ["id", "dataset", "num"]

        num_cols_with_flag = ["oldpeak", "trestbps", "thalch"]
        num_cols_without_flag = ["age", "chol"]
        cat_cols = ["thal", "restecg"]
        cat_ord_cols = ["ca", "cp", "slope"]
        cat_bin_cols = ["sex", "fbs", "exang"]

        cat_cols.extend(cat_ord_cols)
        cat_cols.extend(cat_bin_cols)

        self._manage_missing_values(
            num_cols_without_flag,
            num_cols_with_flag,
            cat_cols,
            cat_ord_cols,
            cat_bin_cols,
        )

        self._binarize_target()

        if encod_binary_features :
            self._encod_binary_features(cat_bin_cols)  # ["sex", "fbs", "exang"]

        self._to_str(["ca"])  # [ca]

        X, y = self._split_data(dropped_cols=dropped_cols)

        X_train, X_test, y_train, y_test = split_set(X, y, train_size=train_size)

        self._fit_missing_values(
            X_train,
            num_cols_without_flag,
            num_cols_with_flag,
            cat_ord_cols,
            cat_bin_cols
        )

        X_train = self._manage_missing_values(
            X_train,
            num_cols_without_flag,
            num_cols_with_flag,
            cat_cols,
            cat_ord_cols,
            cat_bin_cols
        )

        X_test = self._manage_missing_values(
            X_test,
            num_cols_without_flag,
            num_cols_with_flag,
            cat_cols,
            cat_ord_cols,
            cat_bin_cols
        )

        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        self._build_column_transformer(
            num_cols,
            cat_cols,
            use_scaling,
            cat_encoding
        )

        X_train, X_test, y_train, y_test = split_set(X, y, train_size=train_size)

        return X_train, X_test, y_train, y_test, self.preprocess

def manage_missing_values(df) :
    df = df.copy()

    # Note : flag peut-être inutile si ratio NaN < 5 % dataset.

    # === Variables numériques continues ===

    num_cols_with_flag = ["ca", "oldpeak", "trestbps", "thalch"]
    num_cols_without_flag = ["chol"]

    for col in num_cols_with_flag :
        # ajout d'un flag pour indiquer le manque de valeur (0 ou 1)
        df[f"{col}_missing"] = df[col].isnull().astype(int)

        # remplissage des valeurs manquantes par la médiane
        df[col] = df[col].fillna(df[col].median()).infer_objects(copy=False)

    for col in num_cols_without_flag :
        df[col] = df[col].fillna(df[col].median()).infer_objects(copy=False)

    # === Variables catégorielles ===

    cat_cols = ["thal", "restecg"]

    for col in cat_cols :
        # remplissage des vals manquantes par "Unknown"
        df[col] = df[col].fillna("Unknown").infer_objects(copy=False)

    # === Variables ordinales ===

    ord_cols = ["slope"]

    for col in ord_cols :
        df[f"{col}_missing"] = df[col].isnull().astype(int)

        # imputation par la valeur la plus fréquente
        df[col] = df[col].fillna(df[col].mode()[0]).infer_objects(copy=False)


    # === Variables catégoriques binaires ===

    # Note : pas de flag, car var simple, peu d'information structurelle, mode = suffisant.

    bin_cols = ["fbs", "exang"]

    for col in bin_cols :
        df[col] = df[col].fillna(df[col].mode()[0]).infer_objects(copy=False)

    return df


def binarize_target(df) :
    df = df.copy()

    df["target_binary"] = (df["num"] > 0).astype(int)

    return df


def encod_binary_features(df, binary_features) :
    df = df.copy()

    for col in binary_features :
        values = df[col].unique()
        df[col] = df[col].map({values[0]: 1, values[1]: 0})

    return df


def to_str(df, col) :
    df = df.copy()
    df[col] = df[col].astype("str")

    return df


def split_data(df, target=None, dropped_cols: list = None, used_cols: list = None) :
    df = df.copy()

    if dropped_cols is not None : df = df.drop(columns=dropped_cols)
    if used_cols is not None :
        if target is not None : used_cols.append(target)
        df = df[used_cols]

    y = df[target] if target is not None else None
    X = df.drop(columns=target) if target is not None else df

    return X, y


def build_preprocess(num_cols, cat_cols, use_scaling: bool = False, cat_encoding: str = "one_hot") :
    num_transformer = StandardScaler() if use_scaling else "passthrough"

    if cat_encoding == "one_hot" :
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
    elif cat_encoding == "ordinal" :
        cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else :
        cat_transformer = "passthrough"

    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ],
    remainder="drop")


def data_preprocessing(df, target: str = None, dropped_cols: list = None, num_cols = None, cat_cols = None, use_scaling: bool = False, cat_encoding: str = "one_hot") :
    df = manage_missing_values(df)
    df = binarize_target(df)
    df = encod_binary_features(df, ["sex", "fbs", "exang"])
    df = to_str(df, "ca")

    X, y = split_data(df, target, dropped_cols)

    if num_cols is not None :
        num_cols = num_cols.copy()
    else :
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if cat_cols is not None :
        cat_cols = cat_cols.copy()
    else :
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = build_preprocess(num_cols, cat_cols, use_scaling, cat_encoding=cat_encoding)

    return X, y, preprocess


def data_preprocessing_dl(df, target: str = None, dropped_cols: list = None, num_cols = None, cat_cols = None, use_scaling: bool = False, cat_encoding: str = "one_hot") :
    df = manage_missing_values(df)
    df = binarize_target(df)
    df = encod_binary_features(df, ["sex", "fbs", "exang"])
    df = to_str(df, "ca")

    X, y = split_data(df, target, dropped_cols)

    if num_cols is not None :
        num_cols = num_cols.copy()
    else :
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if cat_cols is not None :
        cat_cols = cat_cols.copy()
    else :
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = build_preprocess(num_cols, cat_cols, use_scaling, cat_encoding)

    return X, y, preprocess
"""
