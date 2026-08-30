import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        n_layers = config["n_layers"]

        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config["dropout"])
            )
            for _ in range(n_layers)
        ])

        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_proj(x)

        for block in self.blocks :
            # x = x + block(x)
            x = block(x)

        return self.out_layer(x)

