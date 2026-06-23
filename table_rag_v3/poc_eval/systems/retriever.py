"""
Gateway-embedding + FAISS (CPU) retriever.

Mirrors the repo's `tools/retriever.MixedDocRetriever`:
  * load both modalities -- tables rendered to Markdown (excel_to_markdown) AND the
    per-page document text -- into one corpus;
  * chunk with RecursiveCharacterTextSplitter(1000, 200), prefixing each chunk with
    "File name: {key}\n" so provenance survives;
  * embed + index with faiss IndexFlatIP and search by inner product.

Two deliberate departures from the repo, both consequences of the gateway design:
  * embeddings come from `llm_gateway.embed` instead of a local bge-m3 on GPU;
  * there is no separate cross-encoder reranker (the gateway exposes none), so
    `retrieve` returns the top `rerank_num` by embedding score. The method signature
    is kept identical so the agent code reads the same as the repo.

Embeddings are cached to disk per (version, system) so re-runs skip re-embedding.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from poc_eval import config
from poc_eval.common import llm_gateway
from poc_eval.systems.markdown import excel_to_markdown


class GatewayRetriever:
    def __init__(self, version: str, system_tag: str = "tablerag", include_tables: bool = True):
        self.version = version
        self.include_tables = include_tables
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        save_path = config.embedding_path(version, system_tag)
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                blob = pickle.load(f)
            self.chunks = blob["chunks"]
            self.chunk_file = blob["chunk_file"]
            embeddings = blob["embeddings"]
        else:
            self.chunks, self.chunk_file = self._build_chunks()
            embeddings = self._embed(self.chunks)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "chunk_file": self.chunk_file,
                             "embeddings": embeddings}, f)
        self.index = self._build_index(embeddings)

    # --- corpus construction ----------------------------------------------------------
    def _load_corpus(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        if self.include_tables:
            xdir = config.excel_dir(self.version)
            for fname in sorted(os.listdir(xdir)):
                if fname.endswith(".xlsx"):
                    try:
                        docs[fname] = excel_to_markdown(os.path.join(xdir, fname))
                    except Exception:
                        continue
        ddir = config.doc_dir(self.version)
        for fname in sorted(os.listdir(ddir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(ddir, fname), encoding="utf-8") as fin:
                data = json.load(fin)
            docs[fname] = "".join(f"{k} {v}\n" for k, v in data.items())
        return docs

    def _build_chunks(self) -> Tuple[List[str], List[str]]:
        corpus = self._load_corpus()
        chunks, chunk_file = [], []
        for key, text in corpus.items():
            for split in self.splitter.split_text(text):
                chunks.append(f"File name: {key}\n" + split)
                chunk_file.append(key)
        return chunks, chunk_file

    # --- embedding / index ------------------------------------------------------------
    @staticmethod
    def _embed(texts: List[str]) -> np.ndarray:
        vecs = llm_gateway.embed(texts)
        return np.asarray(vecs, dtype="float32")

    @staticmethod
    def _build_index(embeddings: np.ndarray):
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        return index

    # --- public API -------------------------------------------------------------------
    def retrieve(self, query: str, recall_num: int = config.RECALL_NUM,
                 rerank_num: int = config.RERANK_NUM) -> Tuple[List[str], List[float], List[str]]:
        q = self._embed([query])
        scores, idxs = self.index.search(np.ascontiguousarray(q), min(recall_num, len(self.chunks)))
        docs = [self.chunks[i] for i in idxs[0]]
        files = [self.chunk_file[i] for i in idxs[0]]
        score_list = [float(s) for s in scores[0]]
        return docs[:rerank_num], score_list[:rerank_num], files[:rerank_num]
