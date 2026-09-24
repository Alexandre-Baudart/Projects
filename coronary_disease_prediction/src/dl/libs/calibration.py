import torch.nn as nn

class SigmoidCalib(nn.Module):
    """
    Sigmoid Calibration Layer
    """
    def __init__(self, n_classes):
        super().__init__()

        self.logreg = nn.Linear(n_classes, 1)

        nn.init.ones_(self.logreg.weight)
        nn.init.zeros_(self.logreg.bias)

    def forward(self, logits):
        return self.logreg(logits)

class CalibratedModel(nn.Module):
    def __init__(self, trained_model, config):
        super().__init__()

        self.config = config

        self.trained_model = trained_model
        self.calibrator = self._get_calibrator()

        # trained model is frozen
        self.trained_model.requires_grad_(False)

        # calibrator can be trained
        self.calibrator.requires_grad_(True)

    def _get_calibrator(self):
        if self.config.get("method", None) == "sigmoid":
            return SigmoidCalib(n_classes=self.config["n_classes"])
        else:
            raise RuntimeError(f"Unknown calibration method: {self.config['method']}")

    def train(self, mode=True):
        super().train(mode)

        self.trained_model.eval()
        self.calibrator.train(mode)

        return self

    def forward(self, x):
        logits = self.trained_model(x)
        return self.calibrator(logits)
