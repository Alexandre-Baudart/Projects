from .libs.utils import *
from data.dataset import Dataset_

from .models.svm_kernel import KernelSVM
from .models.logreg import LogisticReg
from .models.random_forest import RandomForest
from .models.xgb import XGBoost
from .models.catb import CatBoost

from .libs.session import MLOrchestrator

METADATA = {
    "name": "project",
    "paths": {
        "metadata": "project_metadata.json",
        "runs": "runs/project/ml"
    },
    "default": {
        "format": "joblib"
    },
    "last_run": 0
}

config = {
    "mode": "clf_binary",

    "optim": {
        "metric": "auc",
        "search_set_size": 0.3
    },

    "cv": {
        "metric": "auc",
        "n_splits": 5
    },

    "train": {
        "metric": "auc"
    },

    "test": {
        "metrics": ["acc", "precision", "recall", "f1-score", "auc", "pr_auc"],
        "calib_eval": False
    },

    "benchmark": {
        "n_iterations": 100
    }
}

model_config = {
    "logreg": {
        "raw_model": LogisticReg,

        "params": {
            'C': 0.64,
            "tol": 1e-4
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "svm_kernel": {
        "raw_model": KernelSVM,

        "params": {
            'C': 3.62,
            "sigma": 5.08,
            "degree": 2,
            "coef0": 1
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "random_forest": {
        "raw_model": RandomForest,

        "params": {
            "n_estimators": 499,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_samples": 0.57
        },

        "preprocess": {
            "use_scaling": False
        }
    },

    "xgb": {
        "raw_model": XGBoost,

        "params": {
            "n_estimators": 1396,
            "learning_rate": 0.08,
            "max_depth": 9,
            "min_child_weight": 1,
            "subsample": 0.82,
            "colsample_bytree": 0.97,
            "reg_alpha": 9e-2,
            "reg_lambda": 1.13
        },

        "preprocess": {
            "use_scaling": False
        }
    },

    "catb": {
        "raw_model": CatBoost,

        "params": {
            "n_estimators": 2729,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 1.43,
        },

        "preprocess": {
            "use_scaling": False,
            "cat_encoding": None
        }
    }
}

if __name__ == "__main__" :
    random_init(seed=42)

    dataset = Dataset_(train_size=0.8)
    dataset.load_csv(data_path="data/heart_disease_uci.csv", dropped_cols=["id", "dataset"])
    dataset.binarize_target(old_target="num", new_target="target_binary", bin_threshold=0)

    session_args = {
        "model_config": model_config,
        "dataset": dataset,
        "config": config,
        "target": "target_binary",
        "metadata": METADATA
    }

    orch = MLOrchestrator(
        session_args=session_args,
    )

    orch.run()


