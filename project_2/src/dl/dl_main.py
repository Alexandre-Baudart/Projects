from data.dataset import Dataset_
from nets import MLP, TabTransformerV1, TabTransformerV2
from .libs.utils import random_init
from ..orchestrator import DLOrchestrator

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

config = {
    "mode": "clf_binary",

    "criterion_info": {
        "type": "bce_logits",
    },

    "optimizer_info": {
        "type": "adamw",
        "lr": 1e-3,
    },

    "lr": 1e-3,
    "batch_size": 64,

    "dataloader_params": {
        "num_workers": 0,
    },

    "metric_info": {
        "metric": "auc",
        "task": "binary"
    },

    "optim_info": {
      "search_set_size": 0.3,
      "max_concurrent_trials": 3
    },

    "train_info": {
        "n_epochs": 50,

        "scheduler_info": {
            "type": "reduce_lr_on_plateau",
            "patience": 3
        },

        "callbacks_info": {
            "early_stopping_params": {
                "patience": 6
            }
        },

        "clip_grad_norm": True,
        "use_stratified_split": False,
    },

    "test_info": {
        "metrics": ["acc", "precision", "recall", "f1-score", "auc", "pr_auc"],
    }
}

model_config = {
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
            "weight_decay": 0.029,

            "params": {
                "n_layers": 4,
                "dropout": 0.23,
                "hidden_size": 128
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
                "dropout": 0.23
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
                "n_frequencies": 8,
                "emb_mlp_hidden_dim": 32,
                "n_heads": 8,
                "dim": 32,
                "n_layers": 3,
                "mlp_hidden_dim": 128,
                "dropout": 0.21,
            }
        },

        "preprocess": {
            "use_scaling": True
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
        "target": "target_binary"
    }

    orch = DLOrchestrator(
        session_args=session_args,
    )

    orch.run()

