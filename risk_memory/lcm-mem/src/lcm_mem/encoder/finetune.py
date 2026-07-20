"""Phase 2: contrastive fine-tuning of the sentence encoder with explicit hard
negatives (MultipleNegativesRankingLoss). Only run if Phase 1 >= weak pass.

Regression guard lives in the caller: evaluate before/after on a small MTEB
retrieval slice and on the Phase 1 discrimination test.
"""

from __future__ import annotations

from pathlib import Path

from lcm_mem.encoder.embed import pick_device


def finetune_encoder(
    train_examples: list[tuple[str, str, list[str]]],
    base_model: str = "intfloat/e5-base-v2",
    out_dir: str | Path = "models/e5-ft",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    seed: int = 0,
) -> str:
    """train_examples: (anchor, positive, [hard_negatives...]).

    Uses sentence-transformers' native trainer; anchor/positive get the e5
    prefixes applied by the caller if needed (we train on raw text since the
    pairs are symmetric declarative statements).
    """
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    torch.manual_seed(seed)
    device = pick_device()
    model = SentenceTransformer(base_model, device=device)

    # MNRL with explicit hard negatives: dataset columns (anchor, positive,
    # negative). Items with multiple negatives are expanded to one row each.
    rows = {"anchor": [], "positive": [], "negative": []}
    for anchor, positive, negatives in train_examples:
        for neg in negatives or [""]:
            if not neg:
                continue
            rows["anchor"].append(anchor)
            rows["positive"].append(positive)
            rows["negative"].append(neg)
    ds = Dataset.from_dict(rows)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        seed=seed,
        use_mps_device=device == "mps",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    final_dir = str(Path(out_dir) / "final")
    model.save(final_dir)
    return final_dir
