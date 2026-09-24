import torch
import torch.nn as nn

class TSLayer(nn.Module):
    """
    Temperature Scaling Layer
    """

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T

class MLSLayer(nn.Module):
    """
    Monotonic Logit Scaling Layer
    """
    def __init__(self, n_classes, alpha_max=2.0):
        super().__init__()

        self.theta = nn.Parameter(torch.zeros(n_classes))
        self.alpha_max = alpha_max

    def forward(self, logits):
        scale = torch.exp( # exp ensures to keep the logits sign
            self.alpha_max * torch.tanh(self.theta)
        )

        return logits * scale

def get_scaling_layer(scaling_info: dict | None = None, **kwargs):
    if scaling_info is not None:
        method = scaling_info.get("method", None)

        if method == "temp":
            return TSLayer()
        elif method == "mls":
            return MLSLayer(n_classes=kwargs["n_classes"], alpha_max=scaling_info.get("alpha_max", 2.0))

    return None