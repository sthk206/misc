"""Predictor shapes, normalization, and a training smoke test on a synthetic
compositional task the MLP can solve but mean-pooling cannot."""

import numpy as np
import pytest
import torch

from lcm_mem.predictor.model import (
    AttnPredictor,
    LearnedPool,
    MLPPredictor,
    MeanPool,
    build_predictor,
    count_params,
)
from lcm_mem.predictor.train import (
    TrainConfig,
    TripleEmbDataset,
    infonce_loss,
    retrieval_mrr,
    train_predictor,
)

DIM = 32


@pytest.mark.parametrize("name", ["mean_pool", "learned_pool", "mlp", "attn"])
def test_output_shape_and_norm(name):
    model = build_predictor(name, dim=DIM)
    a, b, q = (torch.randn(8, DIM) for _ in range(3))
    out = model(a, b, q)
    assert out.shape == (8, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-4)


def test_query_ablation_flag():
    m = MLPPredictor(dim=DIM, use_query=False)
    a, b = torch.randn(4, DIM), torch.randn(4, DIM)
    assert m(a, b, None).shape == (4, DIM)
    with pytest.raises(ValueError):
        MLPPredictor(dim=DIM, use_query=True)(a, b, None)


def test_param_counts_in_spec_range():
    # predictor must stay small (< 20M params per the plan)
    assert count_params(MeanPool()) == 0
    assert count_params(LearnedPool(768)) == 2
    assert 0 < count_params(MLPPredictor(dim=768)) < 20_000_000
    assert 0 < count_params(AttnPredictor(dim=768)) < 20_000_000


def test_infonce_with_hard_negatives_decreases_for_better_preds():
    torch.manual_seed(0)
    target = torch.nn.functional.normalize(torch.randn(16, DIM), dim=-1)
    hard = torch.nn.functional.normalize(torch.randn(16, 4, DIM), dim=-1)
    good = infonce_loss(target, target, hard)
    bad = infonce_loss(-target, target, hard)
    assert good < bad


def _synthetic_npz(path, n, seed):
    """Composition = permuted sum of a and b (nonlinear enough that raw mean
    pooling misses the permutation)."""
    rng = np.random.default_rng(seed)
    perm = np.random.default_rng(123).permutation(DIM)
    a = rng.standard_normal((n, DIM)).astype(np.float32)
    b = rng.standard_normal((n, DIM)).astype(np.float32)
    q = rng.standard_normal((n, DIM)).astype(np.float32)
    t = (a + b)[:, perm].astype(np.float32)
    np.savez(path, a=a, b=b, q=q, target=t)
    return path


def test_mlp_beats_mean_pool_on_learnable_composition(tmp_path):
    train_npz = _synthetic_npz(tmp_path / "train.npz", 512, seed=0)
    val_npz = _synthetic_npz(tmp_path / "val.npz", 128, seed=1)
    cfg = TrainConfig(predictor="mlp", loss="cosine", hidden=128, n_layers=2,
                      batch_size=64, max_epochs=30, patience=5, device="cpu")
    out = train_predictor(train_npz, val_npz, cfg)
    val_ds = TripleEmbDataset(val_npz)
    mean_mrr = retrieval_mrr(MeanPool(), val_ds, "cpu")
    assert out["best_val_mrr"] > mean_mrr + 0.2


def test_checkpoint_roundtrip(tmp_path):
    from lcm_mem.predictor.train import load_predictor

    train_npz = _synthetic_npz(tmp_path / "t.npz", 128, seed=0)
    val_npz = _synthetic_npz(tmp_path / "v.npz", 64, seed=1)
    ckpt = tmp_path / "m.pt"
    cfg = TrainConfig(predictor="mlp", hidden=64, n_layers=2, batch_size=32,
                      max_epochs=2, device="cpu")
    train_predictor(train_npz, val_npz, cfg, ckpt_path=ckpt)
    model = load_predictor(ckpt, device="cpu")
    a, b, q = (torch.randn(3, DIM) for _ in range(3))
    assert model(a, b, q).shape == (3, DIM)
