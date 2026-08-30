from .dataset import ProjectTrainDataset, ProjectTestDataset
import pandas as pd
from project_utils import split_data, split_set, compute_pos_weight
from .nets import MLP

from ray import tune

from .libs.nncore.nno import NNO
from .libs.nncore.nnp import NNP
from .libs.utils import criterion_config, optimizer_config, select_gpu_device, random_init
from .libs.callbacks import callbacks_config

def optimize(X_tr, y_tr, X_te, y_te):
    X_search, _, y_search, _ = split_set(X_tr, y_tr, train_size=0.3)
    dataset_optim = ProjectTrainDataset(X_search, y_search)

    n_classes = dataset_optim.get_n_classes()

    nno = NNO(
        model,
        dataset_train=dataset_optim,
        n_epochs=epochs,
        batch_size=batch_size,
        criterion=criterion,
        metric_info=metric_info,
        dataloader_params=dataloader_params,
    )

    search_space = {
        "n_classes": n_classes,
        "input_size": dataset_optim.get_input_size(),
        "n_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0, 0.3),
        "hidden_size": tune.choice([32, 64, 128]),
    }

    nno.optimize(search_space=search_space, optim_info=optim_info, lr=lr, max_concurrent_trials=3,
                 use_stratified_split=True)

def train_test(X_tr, y_tr, X_te, y_te):
    dataset_train = ProjectTrainDataset(X_tr, y_tr)
    mean, std = dataset_train.get_mean_std()
    dataset_test = ProjectTestDataset(X_te, y_te, mean=mean, std=std)

    # pos_weight = compute_pos_weight(dataset_train)

    n_classes = dataset_train.get_n_classes()

    input_dim = dataset_train.X.shape[1]

    config = {
        "n_layers": 2,
        "hidden_size": 64,
        "dropout": 0.18,
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

    nnp = NNP(final_model, batch_size=batch_size, criterion=criterion, metric_info=metric_info,
              dataloader_params=dataloader_params)
    nnp.train(dataset_train=dataset_train, n_epochs=epochs, lr=lr, optimizer=optimizer, clip_grad_norm=True,
              use_stratified_split=True, callbacks=callbacks, save_best_model=True, save_root="../runs/mlp/")
    # nnp.show_history(show_lr=True)

    # nnp, _ = NNP.load_model(final_model, model_path="../runs/mlp/best_model.pt", criterion=criterion, dataloader_params=dataloader_params, metric_info=metric_info)
    nnp.evaluate(dataset=dataset_test, show_conf_matrix=True, probs_threshold=0.5)


if __name__ == "__main__" :
    random_init(seed=42)

    df = pd.read_csv("./data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df, target="Class", train_size=0.8)

    # device = select_gpu_device()
    model = MLP

    lr = 1e-3
    batch_size = 64
    epochs = 40

    criterion_info = {
        "type": "bce_logits",
        # "pos_weight": pos_weight
    }

    criterion = criterion_config(mode="clf", criterion_info=criterion_info)

    dataloader_params = {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True
    }

    metric_info = {
        "metric": "pr_auc",
        "task": "binary"
    }

    optim_info = {
        "type": "adamw",
        "lr": lr
    }

    # optimize(X_train, y_train, X_test, y_test)
    # train_test(X_train, y_train, X_test, y_test)


