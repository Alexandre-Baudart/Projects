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
    "random_seed": 42,
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
        "calib_eval": True
    },

    "benchmark": {
        "n_iterations": 100
    }
}

model_config = {
    "logreg": {
        "raw_model": LogisticReg,

        "params": {
            'C': 0.74,
            "tol": 2e-4
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "svm_kernel": {
        "raw_model": KernelSVM,

        "params": {
            "kernel": "rbf",
            'C': 4.19,
            "sigma": 3.9
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "random_forest": {
        "raw_model": RandomForest,

        "params": {
            "n_estimators": 307,
            "max_depth": 9,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_samples": 0.70
        },

        "preprocess": {
            "use_scaling": False
        }
    },

    "xgb": {
        "raw_model": XGBoost,

        "params": {
            "n_estimators": 900,
            "learning_rate": 0.09,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.80,
            "colsample_bytree": 0.84,
            "reg_alpha": 0.11,
            "reg_lambda": 0.01
        },

        "preprocess": {
            "use_scaling": False
        }
    },

    "catb": {
        "raw_model": CatBoost,

        "params": {
            "n_estimators": 1881,
            "learning_rate": 0.07,
            "depth": 4,
            "l2_leaf_reg": 0.02,
            "cat_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        },

        "preprocess": {
            "use_scaling": False,
            "cat_encoding": None
        }
    }
}

if __name__ == "__main__" :
    # random_init(seed=42)

    dataset = Dataset_(
        data_path="data/datasets/heart_disease_uci.csv",
        train_path="data/datasets/heart_disease_uci_train.csv",
        calib_path="data/datasets/heart_disease_uci_calib.csv",
        test_path="data/datasets/heart_disease_uci_test.csv",
        dropped_cols=["id", "dataset"],
        binarization_info={
            "old_target": "num",
            "new_target": "target_binary",
            "bin_threshold": 0
        }
    )
    dataset.split_data_csv(train_size=0.8, calib_size=0.1, target="target_binary")

    session_args = {
        "model_config": model_config,
        "dataset": dataset,
        "config": config,
        "target": "target_binary",
        "metadata": None
    }

    orch = MLOrchestrator(
        session_args=session_args,
    )

    orch.run()


