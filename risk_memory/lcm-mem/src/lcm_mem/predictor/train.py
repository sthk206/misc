"""Predictor training on precomputed embeddings.

Losses (run both, report both):
  cosine : 1 - cos(pred, target); targets come from a frozen encoder, so no
           collapse is possible.
  infonce: in-batch negatives (other composed targets) plus optional mined hard
           negatives.
Early stopping on val retrieval MRR (rank the true target among all val targets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lcm_mem.common import set_seed
from lcm_mem.encoder.embed import pick_device
from lcm_mem.predictor.model import build_predictor, count_params


class TripleEmbDataset(Dataset):
    """Precomputed embedding arrays a, b, q, target of shape (N, d); optional
    hard negatives (N, K, d)."""

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path)
        self.a = torch.from_numpy(z["a"]).float()
        self.b = torch.from_numpy(z["b"]).float()
        self.q = torch.from_numpy(z["q"]).float()
        self.t = torch.from_numpy(z["target"]).float()
        self.hard = torch.from_numpy(z["hard"]).float() if "hard" in z else None

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, i: int):
        hard = self.hard[i] if self.hard is not None else torch.empty(0)
        return self.a[i], self.b[i], self.q[i], self.t[i], hard

    @property
    def dim(self) -> int:
        return self.a.shape[1]


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def infonce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hard: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch negatives = other targets; optionally append per-item hard
    negatives (N, K, d)."""
    logits = pred @ target.T / temperature  # (N, N)
    if hard is not None and hard.numel() > 0:
        hard_logits = torch.einsum("nd,nkd->nk", pred, hard) / temperature
        logits = torch.cat([logits, hard_logits], dim=1)
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def retrieval_mrr(
    model: torch.nn.Module, ds: TripleEmbDataset, device: str, batch: int = 512
) -> float:
    """MRR of retrieving each item's target among all targets in `ds`."""
    model.eval()
    targets = F.normalize(ds.t.to(device), dim=-1)
    rr_sum, n = 0.0, len(ds)
    for i in range(0, n, batch):
        a = ds.a[i : i + batch].to(device)
        b = ds.b[i : i + batch].to(device)
        q = ds.q[i : i + batch].to(device)
        pred = model(a, b, q if getattr(model, "use_query", True) else None)
        sims = pred @ targets.T  # (B, N)
        gold_sims = sims[torch.arange(sims.shape[0]), torch.arange(i, i + sims.shape[0])]
        ranks = (sims > gold_sims.unsqueeze(1)).sum(dim=1) + 1
        rr_sum += (1.0 / ranks.float()).sum().item()
    return rr_sum / n


@dataclass
class TrainConfig:
    predictor: str = "mlp"          # mean_pool | learned_pool | mlp | attn
    loss: str = "cosine"            # cosine | infonce
    use_query: bool = True
    hidden: int = 1536
    n_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 5
    temperature: float = 0.05
    seed: int = 0
    device: str = ""
    model_kwargs: dict = field(default_factory=dict)


def train_predictor(
    train_npz: str | Path,
    val_npz: str | Path,
    cfg: TrainConfig,
    ckpt_path: str | Path | None = None,
) -> dict:
    set_seed(cfg.seed)
    device = cfg.device or pick_device()
    train_ds = TripleEmbDataset(train_npz)
    val_ds = TripleEmbDataset(val_npz)

    kwargs: dict = dict(cfg.model_kwargs)
    if cfg.predictor == "mlp":
        kwargs.update(hidden=cfg.hidden, n_layers=cfg.n_layers, use_query=cfg.use_query)
    elif cfg.predictor == "attn":
        kwargs.update(
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            use_query=cfg.use_query,
        )
    model = build_predictor(cfg.predictor, dim=train_ds.dim, **kwargs).to(device)
    n_params = count_params(model)

    history: list[dict] = []
    best_mrr, best_epoch, best_state = -1.0, -1, None

    if n_params == 0:  # MeanPool: nothing to train, just evaluate
        best_mrr = retrieval_mrr(model, val_ds, device)
        best_epoch = 0
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
        )
        for epoch in range(cfg.max_epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for a, b, q, t, hard in loader:
                a, b, q, t = a.to(device), b.to(device), q.to(device), t.to(device)
                hard = hard.to(device) if hard.numel() else None
                pred = model(a, b, q if cfg.use_query else None)
                t_norm = F.normalize(t, dim=-1)
                if cfg.loss == "cosine":
                    loss = cosine_loss(pred, t_norm)
                elif cfg.loss == "infonce":
                    loss = infonce_loss(pred, t_norm, hard, cfg.temperature)
                else:
                    raise ValueError(f"unknown loss {cfg.loss}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            val_mrr = retrieval_mrr(model, val_ds, device)
            history.append(
                {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1),
                 "val_mrr": val_mrr}
            )
            if val_mrr > best_mrr:
                best_mrr, best_epoch = val_mrr, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= cfg.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)

    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "config": cfg.__dict__,
             "dim": train_ds.dim},
            ckpt_path,
        )
    return {
        "n_params": n_params,
        "best_val_mrr": best_mrr,
        "best_epoch": best_epoch,
        "history": history,
        "model": model,
    }


def load_predictor(ckpt_path: str | Path, device: str | None = None) -> torch.nn.Module:
    device = device or pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in TrainConfig.__dataclass_fields__})
    kwargs: dict = {}
    if cfg.predictor == "mlp":
        kwargs.update(hidden=cfg.hidden, n_layers=cfg.n_layers, use_query=cfg.use_query)
    elif cfg.predictor == "attn":
        kwargs.update(d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                      use_query=cfg.use_query)
    model = build_predictor(cfg.predictor, dim=ckpt["dim"], **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
