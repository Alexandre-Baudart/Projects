import matplotlib.pyplot as plt
import os

class History :
    def __init__(self, mode: str) :
        self.train_scores = []
        self.valid_scores = []
        self.train_losses = []
        self.valid_losses = []
        self.lr_list = []

        self.mode = mode
        self.n_epochs = 0

    def save(self, info: dict) :
        self.train_scores.append(info.get("train_score", 0.0))
        self.valid_scores.append(info.get("valid_score", 0.0))
        self.train_losses.append(info.get("train_loss", 0.0))
        self.valid_losses.append(info.get("valid_loss", 0.0))
        self.lr_list.append(info.get("lr", 0.0))

        self.n_epochs += 1

    def rebuild(self, train_scores: list, valid_scores: list, train_losses: list, valid_losses: list, lr_list: list) :
        try :
            assert len(train_scores) == len(valid_scores) == len(train_losses) == len(valid_losses)

            print(train_scores, train_losses, valid_scores, valid_losses, lr_list, sep="\n")

            self.train_scores = train_scores
            self.valid_scores = valid_scores
            self.train_losses = train_losses
            self.valid_losses = valid_losses
            self.lr_list = lr_list

            self.n_epochs = len(self.train_scores)

        except AssertionError :
            print("Impossible to load the history !")


    def show(self, metric: str, show_lr: bool = False, save_fig: bool = False, save_root: str = "./") :
        if self.n_epochs > 1 :
            epochs = [i+1 for i in range(self.n_epochs)]

            n_rows = 1
            if self.mode != "rnn" : n_rows += 1
            if show_lr : n_rows += 1

            fig, axes = plt.subplots(n_rows, 1, figsize=(10, 6))

            if n_rows == 1 :
                axes.plot(epochs, self.train_losses, label="Train")
                axes.plot(epochs, self.valid_losses, label="Validation")
                axes.set_xlabel("Epochs")
                axes.set_ylabel("Loss")
                axes.legend()
            else :
                axes[0].plot(epochs, self.train_losses, label="Train")
                axes[0].plot(epochs, self.valid_losses, label="Validation")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

            if self.mode != "rnn" :
                axes[1].plot(epochs, self.train_scores)
                axes[1].plot(epochs, self.valid_scores)
                if not show_lr: axes[1].set_xlabel("Epochs")
                axes[1].set_ylabel(f"{metric}")

            if show_lr :
                axes[2].plot(epochs, self.lr_list)
                axes[2].set_xlabel("Epochs")
                axes[2].set_ylabel("Learning rate")

            plt.tight_layout()

            if save_fig :
                os.makedirs(save_root, exist_ok=True)
                save_path = save_root + "learning_history.png"
                plt.savefig(save_path)

            plt.show()

