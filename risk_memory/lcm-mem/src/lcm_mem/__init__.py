"""Latent Composition Memory (LCM-Mem).

Two components:
1. A latent composition predictor f(emb(a), emb(b), emb(q)) -> predicted embedding
   of the LLM-composed inference, used as a cheap best-first-search heuristic.
2. A provenance-tracked memory store where derived facts carry parent pointers,
   epistemic types, and confidence, with invalidation propagation.
"""

__version__ = "0.1.0"
