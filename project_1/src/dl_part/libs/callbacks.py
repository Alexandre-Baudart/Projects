import time
import torch
import os

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler


class Callback :
    def on_train_begin(self) : pass
    def on_epoch_begin(self, epoch:int) : pass
    def on_batch_end(self, batch, logs:dict = None) : pass
    def on_epoch_end(self, epoch:int, logs:dict = None) : pass
    def on_train_end(self) : pass


class TrainResultsDisplay(Callback) :
    def on_epoch_end(self, epoch:int, logs:dict = None) :
        train_score = logs.get("train_score", None)
        train_loss = logs.get("train_loss", None)
        valid_score = logs.get("valid_score", None)
        valid_loss = logs.get("valid_loss", None)
        metric = logs.get("metric", None)

        if None in (train_loss, valid_loss) : return
        elif None in (train_score, valid_score) :
            print(f"\n~ Train loss : {train_loss:.4f} - Val loss : {valid_loss:.4f}~")
        else :
            print(f"\n~ Train {metric} : {train_score:.4f} - Val {metric} : {valid_score:.4f} "
                  f"- Train loss : {train_loss:.4f} - Val loss : {valid_loss:.4f} ~")


class TimeMeasuring(Callback) :
    def __init__(self) :
        self.start_time = None

    def on_train_begin(self) :
        self.start_time = time.time()
        print("\nTraining start...\n\n-----")

    def on_train_end(self) :
        elapsed = int(time.time() - self.start_time)

        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        time_parts = []
        if hours > 0:
            time_parts.append(f"{hours}h")
        if minutes > 0:
            time_parts.append(f"{minutes}m")
        time_parts.append(f"{seconds}s")

        print(f"\nTraining finished in {' '.join(time_parts)}.\n")


class EarlyStopping(Callback) :
    def __init__(self, monitor: str = "valid_loss", patience: int = 4, min_delta: float = 1e-4) :
        """
            Params :
                - monitor : str
                - patience : int

            Note :
                The value to monitor must be one of these values : train_score, train_loss, valid_score or valid_loss.

        """
        if monitor not in ("train_loss", "train_score", "valid_loss", "valid_score") :
            self.monitor = "valid_loss"
            print(f"Erreur, \"{monitor}\" ne peut pas être surveillé !\nLa perte en validation (valid_loss) sera surveillée par défaut...\n")
        else :
            self.monitor=monitor

        self.patience = patience
        self.min_delta = min_delta

        self.best = float("inf") if self.monitor in ("train_loss", "valid_loss") else float("-inf")
        self.counter = 0

        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: dict = None) :
        monitored_val = logs[self.monitor]

        if self.monitor in ("train_loss", "valid_loss") :
            improved = monitored_val < self.best - self.min_delta
        else :
            improved = monitored_val > self.best + self.min_delta

        if improved :
            self.best = monitored_val
            self.counter = 0
        else :
            self.counter += 1

        if self.counter > self.patience :
            print(f"\nEarly stopping at epoch {epoch+1}")
            self.stop_training = True


class SchedulerCallback(Callback) :
    def __init__(self, scheduler, optimizer, monitor) :
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.monitor = monitor
        self.current_lr = None

    def state_dict(self) :
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict) :
        self.scheduler.load_state_dict(state_dict)

    def on_epoch_end(self, epoch, logs: dict = None) :
        self.scheduler.step()
        self.current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"\nCurrent learning rate : {self.current_lr}\n\n-----")


class CosineLRSchedulerCallback(Callback) :
    def __init__(self, scheduler, optimizer) :
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.current_lr = None

    def state_dict(self) :
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict) :
        self.scheduler.load_state_dict(state_dict)

    def on_epoch_end(self, epoch, logs: dict = None) :
        self.scheduler.step(epoch)
        self.current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"\nCurrent learning rate : {self.current_lr}\n\n-----")


class PlateauSchedulerCallback(SchedulerCallback) :
    def on_epoch_end(self, epoch: int, logs: dict = None) :
        valid_val = logs.get(self.monitor)

        if valid_val is None :
            return

        self.scheduler.step(valid_val)
        self.current_lr = self.optimizer.param_groups[0]["lr"]

        print(f"\nCurrent learning rate : {self.current_lr}\n\n-----")


class TensorBoardCallback(Callback) :
    def __init__(self, path: str = "runs/exp1") :
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.writer = SummaryWriter(log_dir=path)
        self.step = 0

    def on_epoch_end(self, epoch:int, logs:dict = None) :
        train_score = logs["train_score"]
        train_loss = logs["train_loss"]
        valid_score = logs["valid_score"]
        valid_loss = logs["valid_loss"]

        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_score, epoch)

        self.writer.add_scalar("Loss/valid", valid_loss, epoch)
        self.writer.add_scalar("Accuracy/valid", valid_score, epoch)

    def on_train_end(self, logs=None) :
        self.writer.close()


class BestModelCallback(Callback) :
    def __init__(self, model: torch.nn.Module, optimizer, batch_size: int, scheduler = None, history = None, model_ema = None, save_root: str = "./runs/") :
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.history = history
        self.model_ema = model_ema

        self.save_root = save_root

        self.last_epoch = 0
        self.best_valid_loss = float("inf")
        self.best_model = None
        self.best_lr = self.optimizer.param_groups[0]["lr"]


    def save(self, best_model_path, history_path) :
        self.model.load_state_dict(self.best_model)

        save_dict = {
            "last_epoch": self.last_epoch,
            "model": {k: v.cpu() for k, v in self.best_model.items()}, # envoi de tous les poids du modèle sur CPU pour la sauvegarde
            "batch_size": self.batch_size,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
            "lr": self.best_lr
        }

        if self.model_ema is not None :
            save_dict["model_ema"] = self.model_ema.module.state_dict()

        torch.save(save_dict, best_model_path)

        torch.save(dict(train_scores=self.history.train_scores,
                        train_losses=self.history.train_losses,
                        valid_scores=self.history.valid_scores,
                        valid_losses=self.history.valid_losses,
                        lr_list=self.history.lr_list,
                        n_epochs=self.history.n_epochs),
                   history_path)


    def on_epoch_end(self, epoch: int, logs=None) :
        valid_loss = logs.get("valid_loss")

        if valid_loss is None :
            return

        if valid_loss < self.best_valid_loss :
            self.best_valid_loss = valid_loss
            self.best_model = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.best_lr = self.optimizer.param_groups[0]["lr"]
            self.last_epoch = epoch


    def on_train_end(self) :
        if self.best_model is not None :
            os.makedirs(os.path.dirname(self.save_root), exist_ok=True)

            best_model_path = os.path.join(self.save_root, "best_model.pt")
            history_path = os.path.join(self.save_root, "history.pt")

            self.save(best_model_path, history_path)

            print("Best model saved with success !\n")


class CheckpointCallback(Callback) :
    def __init__(self, model: torch.nn.Module, optimizer, batch_size: int, scheduler = None, history = None, model_ema = None, save_root: str = "./runs/") :
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.history = history
        self.model_ema = model_ema

        self.save_root = save_root


    def checkpoint(self, model_path, history_path, epoch, valid_loss) :
        save_dict = {
            "last_epoch": epoch,
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "batch_size": self.batch_size,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_valid_loss": valid_loss,
        }

        if self.model_ema is not None :
            save_dict["model_ema"] = self.model_ema.module.state_dict()

        torch.save(save_dict, model_path)

        torch.save(dict(train_scores=self.history.train_scores,
                        train_losses=self.history.train_losses,
                        valid_scores=self.history.valid_scores,
                        valid_losses=self.history.valid_losses,
                        lr_list=self.history.lr,
                        n_epochs=self.history.n_epochs),
                   history_path)


    def on_epoch_end(self, epoch: int, logs=None) :
        valid_loss = logs.get("valid_loss")

        if valid_loss is None :
            return

        if (epoch + 1) % 5 == 0 :
            checkpoints_dir = os.path.join(self.save_root, "checkpoints")
            checkpoint_path = os.path.join(checkpoints_dir, f"ckpt_epoch{epoch + 1}")
            os.makedirs(checkpoint_path, exist_ok=True)

            model_path = os.path.join(checkpoint_path, "model.pt")
            history_path = os.path.join(checkpoint_path, "history.pt")

            self.checkpoint(model_path, history_path, epoch, valid_loss)

            print(f"\nCheckpoint at epoch {epoch + 1}.\n")


class ProgressiveUnfreezingCallback(Callback) :
    def __init__(self, model, schedule) :
        self.model = model
        self.schedule = schedule
        self.unfrozen = set()

    def freeze_all(self) :
        for param in self.model.parameters() :
            param.requires_grad = False

    def unfreeze_module(self, module) :
        for param in module.parameters() :
            param.requires_grad = True

    def on_train_begin(self) :
        self.freeze_all()

    def on_epoch_begin(self, epoch: int) :
        if epoch not in self.schedule :
            return

        targets = self.schedule[epoch]

        for target in targets :
            if target == "all":
                for p in self.model.parameters() :
                    p.requires_grad = True
                self.unfrozen.add("all")
            else:
                if target in self.unfrozen :
                    continue

                module = dict(self.model.named_modules())[target]
                self.unfreeze_module(module)
                self.unfrozen.add(target)


def callbacks_config(optimizer = None, monitor="valid_loss", scheduler_info: dict = None,
                     early_stopping_params: dict = None, use_tensorboard: bool = False) :
    """
    Params :
        - optimizer : torch.optim
        - scheduler_info : dict
        - early_stopping_params : dict
        - use_tensorboard : bool

    :returns: dict

    Note :
        Schedulers available : reduce_lr_on_plateau, multi_step_lr or exponential_lr, cosine_lr_scheduler
    """

    callbacks = {}
    scheduler_type = scheduler_info.get("type", None)
    scheduler_params = {k: v for k, v in scheduler_info.items() if k != "type"}

    if (optimizer is not None) and scheduler_type in ("reduce_lr_on_plateau", "multi_step_lr", "exponential_lr", "cosine_lr_scheduler") and scheduler_params :
        if scheduler_type == "reduce_lr_on_plateau" :
            scheduler = ReduceLROnPlateau(optimizer=optimizer, **scheduler_params)
            callbacks["scheduler"] = PlateauSchedulerCallback(scheduler=scheduler, optimizer=optimizer, monitor=monitor)
        elif scheduler_type == "cosine_lr_scheduler" :
            scheduler = CosineLRScheduler(optimizer=optimizer, **scheduler_params)
            callbacks["scheduler"] = CosineLRSchedulerCallback(scheduler=scheduler, optimizer=optimizer)
        else :
            if scheduler_type == "multi_step_lr" :
                scheduler = MultiStepLR(optimizer=optimizer, **scheduler_params)
            else :
                scheduler = ExponentialLR(optimizer=optimizer, **scheduler_params)

            callbacks["scheduler"] = SchedulerCallback(scheduler=scheduler, optimizer=optimizer, monitor=monitor)

    if early_stopping_params is not None :
        callbacks["early_stopping"] = EarlyStopping(**early_stopping_params)

    if use_tensorboard :
        callbacks["tensorboard"] = TensorBoardCallback()

    return callbacks