from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking import Chunk


class HybridRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

    def fit(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        emb = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.astype("float32")
        self.embeddings = emb
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def _vector_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("Retriever not fitted")
        q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def _keyword_scores(self, query: str, top_k: int) -> Dict[int, float]:
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Retriever not fitted")
        q = self.vectorizer.transform([query])
        sim = (self.tfidf_matrix @ q.T).toarray().reshape(-1)
        best = np.argsort(sim)[::-1][:top_k]
        return {int(i): float(sim[i]) for i in best if sim[i] > 0}

    @staticmethod
    def expand_query(query: str) -> str:
        expansions = {
            "budget": "fiscal expenditure revenue allocation policy",
            "election": "votes constituency region candidate results",
            "policy": "initiative program reform regulation strategy",
        }
        terms = [query]
        lower = query.lower()
        for key, val in expansions.items():
            if key in lower:
                terms.append(val)
        return " ".join(terms)

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.65) -> List[dict]:
        expanded_query = self.expand_query(query)
        vec = self._vector_scores(expanded_query, top_k=top_k * 3)
        key = self._keyword_scores(expanded_query, top_k=top_k * 3)

        merged: Dict[int, dict] = {}
        for i, s in vec:
            merged.setdefault(i, {"vector": 0.0, "keyword": 0.0})
            merged[i]["vector"] = s
        for i, s in key.items():
            merged.setdefault(i, {"vector": 0.0, "keyword": 0.0})
            merged[i]["keyword"] = s

        if merged:
            max_key = max(v["keyword"] for v in merged.values()) or 1.0
        else:
            max_key = 1.0

        ranked = []
        for idx, scores in merged.items():
            keyword_norm = scores["keyword"] / max_key
            hybrid_score = alpha * scores["vector"] + (1 - alpha) * keyword_norm
            ranked.append(
                {
                    "chunk": self.chunks[idx],
                    "hybrid_score": float(hybrid_score),
                    "vector_score": float(scores["vector"]),
                    "keyword_score": float(scores["keyword"]),
                }
            )
        ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]

    def save(self, processed_dir: Path) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Retriever not fitted")
        faiss.write_index(self.index, str(processed_dir / "faiss.index"))
        np.save(processed_dir / "embeddings.npy", self.embeddings)
        (processed_dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def load(self, processed_dir: Path) -> None:
        chunks_data = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in chunks_data]
        self.embeddings = np.load(processed_dir / "embeddings.npy")
        self.index = faiss.read_index(str(processed_dir / "faiss.index"))
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
        self.tfidf_matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
