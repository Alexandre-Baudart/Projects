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

