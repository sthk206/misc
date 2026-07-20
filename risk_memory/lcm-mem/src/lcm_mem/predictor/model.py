"""Composition predictor architectures.

All models take (emb_a, emb_b, emb_q) batches of shape (N, d) and return an
L2-normalized (N, d) prediction of emb(composed_gt). Query conditioning is a
config switch (`use_query`) — the ablation surface from the plan.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class MeanPool(nn.Module):
    """(emb_a + emb_b) / 2, renormalized. No parameters."""

    use_query = False

    def forward(self, a: torch.Tensor, b: torch.Tensor, q: torch.Tensor | None = None):
        return _norm((a + b) / 2)


class LearnedPool(nn.Module):
    """Scalar-weighted sum of a and b (softmax weights), renormalized."""

    use_query = False

    def __init__(self, dim: int = 768):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, a: torch.Tensor, b: torch.Tensor, q: torch.Tensor | None = None):
        w = torch.softmax(self.logits, dim=0)
        return _norm(w[0] * a + w[1] * b)


class MLPPredictor(nn.Module):
    """concat[a, b, q, a*b, |a-b|] -> MLP (GELU + LayerNorm) -> d, normalized."""

    def __init__(
        self,
        dim: int = 768,
        hidden: int = 1536,
        n_layers: int = 3,
        use_query: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_query = use_query
        n_feats = 5 if use_query else 4
        layers: list[nn.Module] = []
        in_dim = n_feats * dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor, q: torch.Tensor | None = None):
        feats = [a, b, a * b, (a - b).abs()]
        if self.use_query:
            if q is None:
                raise ValueError("MLPPredictor(use_query=True) requires q")
            feats.insert(2, q)
        return _norm(self.net(torch.cat(feats, dim=-1)))


class AttnPredictor(nn.Module):
    """Small transformer over the token sequence [CLS, a, b, q] with learned
    type embeddings; output = projected CLS. ~5-15M params."""

    def __init__(
        self,
        dim: int = 768,
        d_model: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        use_query: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_query = use_query
        self.in_proj = nn.Linear(dim, d_model)
        n_tokens = 4 if use_query else 3  # CLS, a, b, (q)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_emb = nn.Embedding(n_tokens, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor, q: torch.Tensor | None = None):
        toks = [self.in_proj(a), self.in_proj(b)]
        if self.use_query:
            if q is None:
                raise ValueError("AttnPredictor(use_query=True) requires q")
            toks.append(self.in_proj(q))
        x = torch.stack(toks, dim=1)  # (N, T-1, d_model)
        cls = self.cls.expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.type_emb(torch.arange(x.shape[1], device=x.device))
        x = self.encoder(x)
        return _norm(self.out_proj(x[:, 0]))


PREDICTORS = {
    "mean_pool": MeanPool,
    "learned_pool": LearnedPool,
    "mlp": MLPPredictor,
    "attn": AttnPredictor,
}


def build_predictor(name: str, dim: int = 768, **kwargs) -> nn.Module:
    cls = PREDICTORS[name]
    if cls is MeanPool:
        return cls()
    if cls is LearnedPool:
        return cls(dim=dim)
    return cls(dim=dim, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
