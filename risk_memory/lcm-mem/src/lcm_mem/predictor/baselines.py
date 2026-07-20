"""Non-latent baselines for the cost/quality frontier.

CrossEncoderBaseline scores (query, fact_a + fact_b) with a MiniLM
cross-encoder — measures whether a rerank-style scorer beats the latent
predictor at similar latency. LLM scoring lives in memory/compose.py.
"""

from __future__ import annotations

import time

import numpy as np


class CrossEncoderBaseline:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from lcm_mem.encoder.embed import pick_device

            self._model = CrossEncoder(
                self.model_name, device=self._device or pick_device()
            )
        return self._model

    def score_pairs(
        self, query: str, pairs: list[tuple[str, str]], batch_size: int = 64
    ) -> np.ndarray:
        """Score (query, fact_a + ' ' + fact_b) for each candidate pair."""
        inputs = [(query, f"{a} {b}") for a, b in pairs]
        return np.asarray(
            self.model.predict(inputs, batch_size=batch_size, show_progress_bar=False)
        )


def time_scorer(fn, *args, n_repeat: int = 3, **kwargs) -> tuple[object, float]:
    """Run fn, return (result, best wall-clock seconds over n_repeat)."""
    best = float("inf")
    result = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return result, best
