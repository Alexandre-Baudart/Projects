import torch
import torch.nn as nn
import math

from .logits_scalers import get_scaling_layer

class MLP(nn.Module) :
    def __init__(self, config) :
        super().__init__()

        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        n_layers = config["n_layers"]

        self.use_skip_connection = config.get("use_skip_connection", False)

        self.logits_scaler = get_scaling_layer(scaling_info=config.get("scaling_info", None), n_classes=config["n_classes"])

        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.get("dropout", 0.0))
            )
            for _ in range(n_layers)
        ])

        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x) :
        x = self.input_bn(x)
        x = self.input_proj(x)

        for block in self.blocks :
            if self.use_skip_connection :
                x = x + block(x)
            else :
                x = block(x)

        logits = self.out_layer(x)

        if self.logits_scaler is not None :
            return self.logits_scaler(logits)

        return logits

    def freeze_mlp(self):
        for param in self.blocks.parameters():
            param.requires_grad = False # gradient won't be computed for this param => no modification => frozen

        for param in self.out_layer.parameters():
            param.requires_grad = False

    def freeze_calibrator(self):
        for param in self.calibrator.parameters():
            param.requires_grad = False

    def unfreeze_mlp(self):
        for param in self.blocks.parameters():
            param.requires_grad = True

        for param in self.out_layer.parameters():
            param.requires_grad = True

    def unfreeze_calibrator(self):
        for param in self.calibrator.parameters():
            param.requires_grad = True

class TabTransformerV1(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_num = config["n_num"]
        self.n_cat = config["n_cat"]
        cat_cardinalities = config.get("cat_cardinalities", [])

        self.dim = config.get("dim", 32)
        n_layers = config.get("n_layers", 2)
        n_heads = config.get("n_heads", 4)
        mlp_hidden = config.get("mlp_hidden", 128)

        # ----------------------
        # Embeddings catégoriels
        # ----------------------
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, self.dim)
            for card in cat_cardinalities
        ])

        # Feature / position embeddings (IMPORTANT)
        self.feature_embeddings = nn.Parameter(
            torch.randn(self.n_cat, self.dim)
        )

        # ----------------------
        # Transformer encoder
        # ----------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # ----------------------
        # Normalisation numérique
        # ----------------------
        self.num_bn = nn.BatchNorm1d(self.n_num) if self.n_num > 0 else None

        # ----------------------
        # MLP final
        # ----------------------
        if self.n_cat > 0 :
            input_dim = self.dim + self.n_num
        else :
            input_dim = self.n_num

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.0)),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        x_num = x[:, :self.n_num].float()
        x_cat = x[:, self.n_num:].long()

        if self.n_cat > 0:
            cat_tokens = [
                emb(x_cat[:, i])
                for i, emb in enumerate(self.cat_embeddings)
            ]

            x_cat = torch.stack(cat_tokens, dim=1)
            x_cat = x_cat + self.feature_embeddings
            x_cat = self.transformer(x_cat)
            x_cat = x_cat.mean(dim=1)

        if self.n_num > 0 and self.n_cat > 0:
            x = torch.cat([x_cat, x_num], dim=1)
        elif self.n_cat > 0:
            x = x_cat
        else:
            x = x_num

        return self.mlp(x)

class FourierNumericalEmbedding(nn.Module):
    def __init__(self, n_features: int, n_frequencies: int, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()

        self.n_features = n_features
        self.n_frequencies = n_frequencies
        self.embedding_dim = embedding_dim

        if hidden_dim is None:
            hidden_dim = embedding_dim

        # Fréquences utilisées pour chaque variable
        frequencies = torch.arange(
            1,
            n_frequencies + 1,
            dtype=torch.float32
        )

        self.register_buffer("frequencies", frequencies)

        # Fourier features
        # sin(2pifx) + cos(2pifx)
        fourier_dim = 2 * n_frequencies

        # tokenizer MLP
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        # x = [batch, n_features]
        x = x.unsqueeze(-1)

        # [batch, n_features, n_frequencies]
        angles = (
            2
            * math.pi
            * x
            * self.frequencies
        )

        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        # [batch, n_features, 2 * n_frequencies]
        fourier_features = torch.cat(
            [sin_features, cos_features],
            dim=-1
        )

        # MLP appliqué indépendamment à chaque variable
        # [batch, n_features, embedding_dim]
        embeddings = self.mlp(fourier_features)

        return embeddings

class NumericalEmbedding(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()

        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            hidden_dim = embedding_dim

        # tokenizer MLP
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        # x : [batch, n_features]
        x = x.unsqueeze(-1)

        # [batch, n_features, 1]
        embeddings = self.mlp(x)

        # [batch, n_features, embedding_dim]
        return embeddings

class TabTransformerV2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_num = config["n_num"]
        self.n_cat = config["n_cat"]
        cat_cardinalities = config.get("cat_cardinalities", [])

        self.dim = config.get("dim", 32)
        n_layers = config.get("n_layers", 2)
        n_heads = config.get("n_heads", 4)
        mlp_hidden_dim = config.get("mlp_hidden_dim", 64)

        # Embeddings catégoriels
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, self.dim)
            for card in cat_cardinalities
        ])

        # Embeddings numériques
        if self.n_num > 0:
            num_embedding_method = config.get("num_embedding_method", "standard")

            if num_embedding_method == "standard":
                self.num_tokenizer = NumericalEmbedding(
                    n_features=self.n_num,
                    embedding_dim=self.dim,
                    hidden_dim=config.get("emb_mlp_hidden_dim", self.dim)
                )

            elif num_embedding_method == "fourier":
                self.num_tokenizer = FourierNumericalEmbedding(
                    n_features=self.n_num,
                    n_frequencies=config.get("n_frequencies", 8),
                    embedding_dim=self.dim,
                    hidden_dim=config.get("emb_mlp_hidden_dim", self.dim)
                )

            else:
                raise ValueError(f"Unknown numerical embedding method: {num_embedding_method}")

        else:
            self.num_tokenizer = None

        # Feature / position embeddings (IMPORTANT)
        self.feature_embeddings = nn.Parameter(
            torch.randn(self.n_num + self.n_cat, self.dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # ----------------------
        # MLP final
        # ----------------------
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.0)),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, x):
        x_num = x[:, :self.n_num].float()
        x_cat = x[:, self.n_num:].long()

        tokens = []

        # Tokens numériques
        if self.n_num > 0:
            num_tokens = self.num_tokenizer(x_num)
            tokens.append(num_tokens)

        # Tokens catégoriels
        if self.n_cat > 0:
            cat_tokens = [
                emb(x_cat[:, i])
                for i, emb in enumerate(self.cat_embeddings)
            ]

            cat_tokens = torch.stack(
                cat_tokens,
                dim=1
            )

            tokens.append(cat_tokens)

        # Tous les tokens
        x = torch.cat(tokens, dim=1)

        # Feature embeddings
        x = x + self.feature_embeddings

        # Transformer
        x = self.transformer(x)

        # Aggrégation
        x = x.mean(dim=1)

        # Prediction
        return self.mlp(x)