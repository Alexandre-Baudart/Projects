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
                nn.GELU(),
                nn.Dropout(config["dropout"])
            )
            for _ in range(n_layers)
        ])

        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_proj(x)

        for block in self.blocks :
            x = x + block(x)
            x = block(x)

        return self.out_layer(x)


class TabTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cat_idx = config.get("cat_idx", [])
        self.num_idx = config.get("num_idx", [])
        cat_cardinalities = config.get("cat_cardinalities", [])

        dim = config.get("dim", 32)
        n_layers = config.get("n_layers", 2)
        n_heads = config.get("n_heads", 4)
        mlp_hidden = config.get("mlp_hidden", 128)

        # ----------------------
        # Embeddings catégoriels
        # ----------------------
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card in cat_cardinalities
        ])

        # Feature / position embeddings (IMPORTANT)
        self.feature_embeddings = nn.Parameter(
            torch.randn(len(self.cat_idx), dim)
        )

        # ----------------------
        # Transformer encoder
        # ----------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # ----------------------
        # Normalisation numériques
        # ----------------------
        self.num_bn = nn.BatchNorm1d(len(self.num_idx)) if len(self.num_idx) > 0 else None

        # ----------------------
        # MLP final
        # ----------------------
        if len(self.cat_idx) > 0 :
            input_dim = dim + len(self.cat_idx)
        else :
            input_dim = len(self.num_idx)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x) :
        # ----------------------
        # numériques
        # ----------------------
        if len(self.num_idx) > 0 :
            x_num = x[:, self.num_idx].float()
            if self.num_bn is not None :
                x_num = self.num_bn(x_num)
        else:
            x_num = None

        # ----------------------
        # catégoriels + transformer
        # ----------------------
        if len(self.cat_idx) > 0 :
            x_cat = x[:, self.cat_idx].long()

            cat_tokens = [
                emb(x_cat[:, i])
                for i, emb in enumerate(self.cat_embeddings)
            ]

            x_cat = torch.stack(cat_tokens, dim=1)

            # + feature embedding (important)
            x_cat = x_cat + self.feature_embeddings

            # transformer
            x_cat = self.transformer(x_cat)

            # pooling (au lieu de flatten)
            x_cat = x_cat.mean(dim=1)

        else:
            x_cat = None

        # ----------------------
        # concat final
        # ----------------------
        if x_num is not None and x_cat is not None:
            x_final = torch.cat([x_cat, x_num], dim=1)
        elif x_cat is not None:
            x_final = x_cat
        else:
            x_final = x_num

        # ----------------------
        # output logits
        # ----------------------
        return self.mlp(x_final)