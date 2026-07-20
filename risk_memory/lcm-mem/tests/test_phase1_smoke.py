"""Offline end-to-end smoke test of the Phase 1 kill-experiment pipeline using
the hashing embedder and synthetic triples. Verifies the machinery (training,
stratification, discrimination, pruning, verdict), not the science."""

import json
import random

from lcm_mem.evals.kill_experiment import VerdictInputs, decide_verdict
from lcm_mem.evals.run_phase1 import run_phase1

FIRST = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry"]
CITY = ["Paris", "Berlin", "Madrid", "Rome", "Lisbon", "Vienna", "Oslo", "Dublin"]


def _write_triples(path, n, seed):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for i in range(n):
            who = rng.choice(FIRST)
            city = rng.choice(CITY)
            year = rng.randint(1950, 2020)
            row = {
                "qid": f"2hop__{seed}{i}a_{seed}{i}b",
                "group_id": f"g{seed}{i}",
                "hops": 2,
                "fact_a": f"{who} number {i} was born in {city}.",
                "fact_b": f"The mayor of {city} took office in {year}.",
                "composed_gt": f"{who} number {i} was born in the city whose mayor took office in {year}.",
                "query": f"When did the mayor of {who} number {i}'s birth city take office?",
                "answerable": True,
                "pair_depth": 0,
            }
            f.write(json.dumps(row) + "\n")


def test_run_phase1_end_to_end(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    _write_triples(data_dir / "musique_triples.train.jsonl", 60, seed=1)
    _write_triples(data_dir / "musique_triples.val.jsonl", 20, seed=2)
    _write_triples(data_dir / "musique_triples.test.jsonl", 20, seed=3)

    result = run_phase1(
        data_dir=data_dir,
        emb_dir=tmp_path / "emb",
        ckpt_dir=tmp_path / "models",
        embedder_name="hash-32",
        methods=("mean_pool", "mlp"),
        losses=("cosine",),
        query_ablation=False,
        include_cross_encoder=False,   # avoid model download in unit tests
        n_pruning_items=10,
        seed=0,
        results_dir=tmp_path / "results",
        train_overrides={"batch_size": 16, "max_epochs": 3, "patience": 2,
                         "device": "cpu"},
    )
    assert result["verdict"] in ("STRONG PASS", "WEAK PASS", "FAIL")
    assert (tmp_path / "results" / "phase1_verdict.md").exists()
    assert "mlp_cosine" in result["per_method"]
    rm = result["per_method"]["mean_pool"]["retrieval"]
    assert 0.0 <= rm["MRR"] <= 1.0
    assert set(result["per_method"]["mean_pool"]["per_tercile"]) <= {0, 1, 2}
    assert "predictor" in result["pruning"]
    assert result["pruning"]["predictor"]["n_items"] == 10


def test_verdict_gates():
    strong, _ = decide_verdict(VerdictInputs(
        mrr_overall={"predictor": 0.5, "mean_pool": 0.3},
        mrr_low_overlap={"predictor": 0.45, "mean_pool": 0.2},
        pruning={
            "predictor": {"gold_recall@5": 0.9, "gold_recall@10": 0.95,
                           "latency_ms_per_1000_pairs": 5, "n_pairs": 1900,
                           "n_items": 10},
            "cross_encoder": {"gold_recall@5": 0.85, "gold_recall@10": 0.9,
                               "latency_ms_per_1000_pairs": 500, "n_pairs": 1900,
                               "n_items": 10},
        },
    ))
    assert strong == "STRONG PASS"

    weak, _ = decide_verdict(VerdictInputs(
        mrr_overall={"predictor": 0.32, "mean_pool": 0.3},
        mrr_low_overlap={"predictor": 0.22, "mean_pool": 0.2},
        pruning={
            "predictor": {"gold_recall@5": 0.7, "gold_recall@10": 0.93,
                           "latency_ms_per_1000_pairs": 5, "n_pairs": 1900,
                           "n_items": 10},
            "cross_encoder": {"gold_recall@5": 0.95, "gold_recall@10": 0.99,
                               "latency_ms_per_1000_pairs": 500, "n_pairs": 1900,
                               "n_items": 10},
        },
    ))
    assert weak == "WEAK PASS"

    fail, _ = decide_verdict(VerdictInputs(
        mrr_overall={"predictor": 0.31, "mean_pool": 0.3},
        mrr_low_overlap={"predictor": 0.2, "mean_pool": 0.2},
        pruning={
            "predictor": {"gold_recall@5": 0.5, "gold_recall@10": 0.6,
                           "latency_ms_per_1000_pairs": 5, "n_pairs": 1900,
                           "n_items": 10},
            "cross_encoder": {"gold_recall@5": 0.95, "gold_recall@10": 0.99,
                               "latency_ms_per_1000_pairs": 500, "n_pairs": 1900,
                               "n_items": 10},
        },
    ))
    assert fail == "FAIL"
