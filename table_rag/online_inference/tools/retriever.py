"""
retriever.py

POC adaptations (only what the gateway / CPU-only environment forces; flow unchanged):
  - Embeddings come from the gateway via the patched utils.tool_utils.Embedder.
  - FAISS runs on CPU (no StandardGpuResources / GpuClonerOptions).
  - Reranker is a pass-through (utils.tool_utils.Reranker), so recall order is kept.
  - load_hybrid_dataset reads tables as JSON (not .xlsx): each table json ->
    DataFrame -> markdown, matching the repo's "table as markdown chunk" behaviour.
  - Optional embedding pickle cache disabled when save_path is None.
The recall->rerank flow, chunking (1000/200), and "File name:" prefixing are unchanged.
"""

import os
import json
import faiss
import threading
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from utils.tool_utils import Embedder, Reranker


class SemanticRetriever:
    """Retrieving process, containing recall and (pass-through) rerank."""

    def __init__(self, chunks, chunk_index, chunk_file_index=None, llm_path=None,
                 reranker_path=None, save_path=None) -> None:
        self.embedding_model = Embedder(llm_path)
        self.reranker = Reranker(reranker_path)

        if save_path and os.path.exists(save_path):
            doc_embeddings, self.chunks, self.chunk_file_index = self.load_embeddings(save_path)
            self.chunk_index = {idx: ch for idx, ch in enumerate(self.chunks)}
        else:
            self.chunks = chunks
            self.chunk_index = chunk_index
            self.chunk_file_index = chunk_file_index
            doc_embeddings = self.embed_doc(chunks, save_path=save_path)

        self.index_lock = threading.RLock()
        print("embedding size", doc_embeddings.shape)
        self.index_IP = self.build_index(doc_embeddings)

    def embed_doc(self, chunks, batch_size: int = 512, save_path: str = None) -> Any:
        if isinstance(chunks, str):
            chunks = [chunks]
        encode_vecs = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            if len(batch) == 1:
                encode_vecs.append(batch_embeddings)
            else:
                encode_vecs.extend(batch_embeddings)

        encode_vecs = np.array(encode_vecs)
        if len(encode_vecs.shape) == 3:
            encode_vecs = encode_vecs.reshape(-1, encode_vecs.shape[-1])

        if save_path:
            self.save_embeddings(encode_vecs, chunks, save_path)
        return encode_vecs

    def save_embeddings(self, embeddings, chunks, save_path) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.pkl'):
            with open(save_path, "wb") as f:
                pickle.dump({"embeddings": embeddings, "chunks": chunks,
                             "chunk_file_index": self.chunk_file_index}, f)

    @staticmethod
    def load_embeddings(load_path: str = None) -> Tuple[Any, Any, Any]:
        if load_path.endswith('.npy'):
            return np.load(load_path)
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data['chunks'], data['chunk_file_index']

    def build_index(self, dense_vector: Any) -> Any:
        with self.index_lock:
            _, dim = dense_vector.shape
            index_IP = faiss.IndexFlatIP(dim)   # CPU index
            index_IP.add(dense_vector)
            return index_IP

    def retrieve(self, query, recall_num, rerank_num):
        docs, ori_file_name = self.recall(query, recall_num)
        reranked_docs, rerank_scores, filenames = self.rerank(query, docs, rerank_num, ori_file_name)
        return reranked_docs, rerank_scores, filenames

    def recall(self, query: str, topn: int) -> Tuple[List[str], List[str]]:
        query_emb = self.embed_doc(query)
        with self.index_lock:
            D, I = self.index_IP.search(query_emb, min(topn, len(self.chunks)))
        ori_docs = [self.chunk_index[i] for i in I[0]]
        ori_file_name = [self.chunk_file_index[i] for i in I[0]]
        return ori_docs, ori_file_name

    def rerank(self, query, docs, topn, ori_file_name) -> Tuple[List[str], List[int], List[str]]:
        pairs = [[query, d] for d in docs]
        scores = self.reranker.compute_score(pairs)
        sorted_pairs = sorted(zip(scores, docs, ori_file_name), key=lambda x: x[0], reverse=True)
        score_sorted, doc_sorted, filename_sorted = zip(*sorted_pairs)
        return list(doc_sorted[:topn]), list(score_sorted[:topn]), list(filename_sorted[:topn])


class MixedDocRetriever:
    def __init__(self, doc_dir_path, excel_dir_path, llm_path=None, reranker_path=None,
                 save_path=None) -> None:
        self.ori_documents = self.load_hybrid_dataset(doc_dir_path, excel_dir_path)
        print("Loading done.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunking_dict = self.doc_chunking()
        self.chunks, self.chunk_to_index, self.chunk_to_filename = self.build_index(doc_chunking_dict)
        self.semantic_retriever = SemanticRetriever(
            chunks=self.chunks, chunk_index=self.chunk_to_index,
            chunk_file_index=self.chunk_to_filename, llm_path=llm_path,
            reranker_path=reranker_path, save_path=save_path,
        )

    def load_hybrid_dataset(self, doc_dir_path: str, excel_dir_path: str) -> Dict[str, str]:
        all_docs: Dict[str, str] = {}
        # Tables (T): JSON files {"columns":[...], "data":[[...]]} -> markdown.
        if excel_dir_path and os.path.isdir(excel_dir_path):
            for file in os.listdir(excel_dir_path):
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(excel_dir_path, file), encoding="utf-8") as f:
                    t = json.load(f)
                df = pd.DataFrame(t["data"], columns=t["columns"])
                all_docs[file] = f"Table name: {t.get('table_name', file)}\n" + df.to_markdown(index=False)
        # Text docs (D): JSON key-value -> text (as in the repo).
        if doc_dir_path and os.path.isdir(doc_dir_path):
            for file in os.listdir(doc_dir_path):
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(doc_dir_path, file), encoding="utf-8") as f:
                    data_split = json.load(f)
                all_docs[file] = "".join(f"{k} {v}\n" for k, v in data_split.items())
        return all_docs

    def build_index(self, chunking_dict: Dict) -> Tuple:
        flatten_chunks, chunk_to_index, chunk_to_filename = [], defaultdict(list), defaultdict(str)
        cnt = 0
        for key, item in chunking_dict.items():
            flatten_chunks += item
            for i in item:
                chunk_to_index[cnt] = i
                chunk_to_filename[cnt] = key
                cnt += 1
        return flatten_chunks, chunk_to_index, chunk_to_filename

    def nltk_single_doc_chunking(self, doc: str, key: str) -> Tuple[List[str], str]:
        all_splits = self.text_splitter.split_text(doc)
        return [f"File name: {key}\n" + s for s in all_splits], key

    def doc_chunking(self, max_workers=10) -> Dict:
        doc_chunkings = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.nltk_single_doc_chunking, doc, key): key
                       for key, doc in self.ori_documents.items()}
            for future in as_completed(futures):
                splits, key = future.result()
                doc_chunkings[key] += splits
        return doc_chunkings

    def retrieve(self, query: str, recall_num: int = 50, rerank_num: int = 5):
        return self.semantic_retriever.retrieve(query, recall_num, rerank_num)
