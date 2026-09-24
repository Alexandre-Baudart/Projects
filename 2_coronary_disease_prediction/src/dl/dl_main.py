from pydoc import Helper

from data.dataset import Dataset_
from .libs.nets import MLP, TabTransformerV1, TabTransformerV2
from .libs.session import DLOrchestrator

from ray import tune

METADATA = {
    "name": "project",
    "paths": {
        "metadata": "project_metadata.json",
        "runs": "runs/project/dl"
    },
    "default": {
        "format": "pt"
    },
    "last_run": 0
}

HELPER = """
DL Session Helper

==================

Usage:
    python -m src.dl.dl_main [OPTIONS]

Available models:
    mlp -- MLP
    tab-transformer-v1 -- Tab-Transformer (V1)
    tab-transformer-v2 -- Tab-Transformer (V2)

Available actions:
    optimize
    train
    test -- requires train or model_path
    check_high_confidence_bias -- requires train or model_path
    benchmark -- requires train or model_path
    
Available devices:
    cpu
    gpu (if accessible)

Options:
    -m, --model <model>:
        to select a model for the session
        
    -lsc, --logits_scaling:
        to scale the logits

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
        to specify a save format for model (default: pt)
        
    -l, --load <model_path>:
        to load the specified model

    -cm, --conf_matrix:
        to enable the building of confusion matrix

    -d, --device:
        to change the execution device (default: cpu)

    -cs, --clean_sandbox:
        to clean sandbox
    
    -i, --info:
        show this information message and exit
"""

CONFIG = {
    "random_seed": 42,
    "mode": "clf_binary",

    "criterion_info": {
        "type": "bce_logits",
    },

    "lr": 1e-3,
    "batch_size": 64,

    "dataloader_params": {
        "num_workers": 0,
    },

    "callbacks_info": {
        "early_stopping_params": {
            "patience": 6
        }
    },

    "optim_info": {
        "search_set_size": 0.3,
        "max_concurrent_trials": 3,

        "metric_info": {
            "metric": "auc",
            "task": "binary"
        },

        "optimizer_info": {
            "type": "adamw",
            "lr": 1e-3,
        }
    },

    "train_info": {
        "n_epochs": 50,

        "metric_info": {
            "metric": "auc",
            "task": "binary"
        },

        "optimizer_info": {
            "type": "adamw",
            "lr": 1e-3,
        },

        "scheduler_info": {
            "type": "reduce_lr_on_plateau",
            "patience": 3
        },

        "clip_grad_norm": True,
        "use_stratified_split": False,
    },

    "calib_info": {
        "method": "sigmoid",
        "ratio": 0.2,

        "metric_info": {
            "metric": "brier_score",
            "task": "binary"
        },

        "optimizer_info": {
            "type": "sgd",
            "lr": 1e-3,
        }
    },

    "test_info": {
        "metrics": ["acc", "precision", "recall", "f1_score", "auc", "pr_auc"],
        "calib_eval": True
    }
}

MODEL_CONFIG = {
    "mlp": {
        "raw_model": MLP,

        "get_transformer_stuff": False,

        "optim": {
            "search_space": {
                "n_layers": tune.choice([2, 3, 4]),
                "dropout": tune.uniform(0, 0.3),
                "hidden_size": tune.choice([32, 64, 128]),
                "weight_decay": tune.loguniform(1e-3, 1e-1)
            }
        },

        "train": {
            "weight_decay": 0.001,

            "params": {
                "n_layers": 4,
                "dropout": 0.13,
                "hidden_size": 128,
                # "scaling_info": { "method": "mls" },
            }
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "tab-transformer-v1": {
        "raw_model": TabTransformerV1,

        "get_transformer_stuff": True,

        "optim": {
            "search_space": {
                "n_heads": tune.choice([4, 8]),
                "dim": tune.choice([32, 64, 128]),
                "n_layers": tune.choice([2, 3]),
                "mlp_hidden_dim": tune.choice([64, 128]),
                "dropout": tune.uniform(0, 0.3),
                "weight_decay": tune.loguniform(1e-3, 1e-1)
            }
        },

        "train": {
            "weight_decay": 0.015,

            "params": {
                "n_heads": 4,
                "dim": 32,
                "n_layers": 2,
                "mlp_hidden": 64,
                "dropout": 0.23,
            }
        },

        "preprocess": {
            "use_scaling": True
        }
    },

    "tab-transformer-v2": {
        "raw_model": TabTransformerV2,

        "get_transformer_stuff": True,

        "optim": {
            "search_space": {
                "n_frequencies": tune.choice([2, 4, 8]),
                "num_embedding_method": "standard",
                "emb_mlp_hidden_dim": tune.choice([32, 64]),
                "n_heads": tune.choice([4, 8]),
                "dim": tune.choice([32, 64, 128]),
                "n_layers": tune.choice([2, 3]),
                "mlp_hidden_dim": tune.choice([64, 128]),
                "dropout": tune.uniform(0, 0.3),
                "weight_decay": tune.loguniform(1e-3, 1e-1)
            }
        },

        "train": {
            "weight_decay": 0.001,

            "params": {
                "num_embedding_method": "standard",
                "n_frequencies": 4,
                "emb_mlp_hidden_dim": 64,
                "n_heads": 8,
                "dim": 32,
                "n_layers": 2,
                "mlp_hidden_dim": 128,
                "dropout": 0.18,
            }
        },

        "preprocess": {
            "use_scaling": True
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

    orch = DLOrchestrator(
        session_args=session_args,
    )

    orch.run()

