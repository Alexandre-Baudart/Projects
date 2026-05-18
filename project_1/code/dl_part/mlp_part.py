from .dataset import ProjectDataset
import pandas as pd
from project_utils import split_data, split_set, compute_pos_weight
from .nets import MLP

from ray import tune

from .libs.nncore.nno import NNO
from .libs.nncore.nnp import NNP
from .libs.utils import criterion_config, optimizer_config, select_gpu_device, random_init
from .libs.callbacks import callbacks_config


if __name__ == "__main__" :
    random_init(seed=42)

    df = pd.read_csv("./data/creditcard.csv")
    X, y = split_data(df, target="Class")

    X_train, X_test, y_train, y_test = split_set(X, y, train_size=0.8)

    # X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
    # X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    dataset_train = ProjectDataset(X_train, y_train)
    dataset_test = ProjectDataset(X_test, y_test)

    n_classes = len(dataset_train.classes) - 1

    # device = select_gpu_device()
    model = MLP

    lr = 1e-3
    batch_size = 64
    epochs = 40

    criterion_info = {
        "type": "bce_logits"
    }

    criterion = criterion_config(mode="clf", criterion_info=criterion_info)

    dataloader_params = {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True
    }

    dataloader_params_optim = {
        "num_workers": 0,
    }

    metric_info = {
        "metric": "pr_auc",
        "task": "binary"
    }

    optim_info = {
        "type": "adamw",
        "lr": lr
    }

    """
    nno = NNO(
        model,
        dataset_train=dataset_train,
        n_epochs=epochs,
        batch_size=batch_size,
        criterion=criterion,
        metric_info=metric_info,
        dataloader_params=dataloader_params_optim,
    )

    search_space = {
        "n_classes": n_classes,
        "input_size": dataset_train.X.shape[1],
        "n_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0, 0.3),
        "hidden_size": tune.choice([32, 64, 128]),
    }

    nno.optimize(search_space=search_space, optim_info=optim_info, lr=lr, max_concurrent_trials=3)
    """

    input_dim = dataset_train.X.shape[1]

    config = {
        "n_layers": 3,
        "hidden_size": 128,
        "dropout": 0.08,
        "n_classes": n_classes,
        "input_size": input_dim,
    }
    
    final_model = MLP(config)

    optimizer = optimizer_config(final_model, optim_info=optim_info)

    scheduler_info = {
        "type": "reduce_lr_on_plateau",
        "patience": 3
    }

    callbacks = callbacks_config(optimizer, scheduler_info=scheduler_info, early_stopping_params=dict(patience=6))

    nnp = NNP(final_model, batch_size=batch_size, criterion=criterion, metric_info=metric_info, dataloader_params=dataloader_params)
    nnp.train(dataset_train=dataset_train, n_epochs=epochs, lr=lr, optimizer=optimizer, clip_grad_norm=True, use_stratified_split=True, callbacks=callbacks, save_best_model=True, save_root="./runs/mlp/")
    # nnp.show_history(show_lr=True)

    # nnp, _ = NNP.load_model(final_model, model_path="./runs/mlp/best_model.pt", criterion=criterion, dataloader_params=dataloader_params, metric_info=metric_info)
    # nnp.evaluate(dataset=dataset_test, show_conf_matrix=True)
    # nnp.to_onnx(input_dim=input_dim, filename="mlp")
