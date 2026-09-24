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

HELPER = """
ML Session Helper

==================

Usage:
    python -m src.ml.ml_main [OPTIONS]
    
Available models:
    logreg -- Logistic Regression
    svm -- SVM
    random_forest -- Random Forest
    xgb -- XGBoost
    catb -- CatBoost
    
Available actions:
    optimize
    train
    cross-validate
    test -- requires train or model_path
    check_high_confidence_bias -- requires train or model_path
    benchmark -- requires train or model_path
    
Available formats:
    joblib
    skops
        
Options:
    -m, --model <model>:
        to select a model for the session
    
    -a, --actions <actions>:
        to resolve the provided actions
        
        Examples:
            -a train
            -a optimize
            -a train test
            -a test -l <model_path> 
    
    -c, --calibrate:
        to calibrate the model
    
    -s, --save:
        to save the session results (metadata, train, test,...) 
        and the model 
    
    -r, --root <new_root>:
        to specify a new root for backup (default: runs/sandbox)
    
    -sn, --save_name <save_name>:
        to specify a save filename (optional)
        
    -sf, --save_format <save_format>:
        to specify a save format for model (default: joblib)
        
    -l, --load <model_path>:
        to load the specified model
    
    -cm, --conf_matrix:
        to enable the building of confusion matrix
    
    -gfi, --get_features_importance:
        to recover features importance
    
    -cs, --clean_sandbox:
        to clean sandbox
        
    -i, --info:
        show this information message and exit
"""

CONFIG = {
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

MODEL_CONFIG = {
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
        "model_config": MODEL_CONFIG,
        "dataset": dataset,
        "config": CONFIG,
        "target": "target_binary",
        "metadata": None,
        "helper": HELPER,
    }

    orch = MLOrchestrator(
        session_args=session_args,
    )

    orch.run()


