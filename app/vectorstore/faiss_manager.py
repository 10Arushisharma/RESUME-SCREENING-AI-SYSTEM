from __future__ import annotations

import faiss
import numpy as np


class FAISSManager:
    """A lightweight wrapper around FAISS for embedding storage and retrieval."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = self.create_index(dimension)
        self.ids: list[str] = []

    def create_index(self, dimension: int):
        return faiss.IndexFlatIP(dimension)

    def reset(self) -> None:
        self.index = self.create_index(self.dimension)
        self.ids.clear()

    def insert(self, vector: np.ndarray, candidate_id: str) -> None:
        self.index.add(np.asarray([vector], dtype=np.float32))
        self.ids.append(candidate_id)

    def search(self, vector: np.ndarray, k: int = 5) -> tuple[list[float], list[int]]:
        if self.index.ntotal == 0:
            return [], []
        distances, indices = self.index.search(np.asarray([vector], dtype=np.float32), min(k, self.index.ntotal))
        return distances[0].tolist(), indices[0].tolist()

    def update(self, candidate_id: str, vector: np.ndarray) -> None:
        if candidate_id not in self.ids:
            self.insert(vector, candidate_id)
            return
        position = self.ids.index(candidate_id)
        self.index.remove_ids(np.array([position], dtype=np.int64))
        self.index.add(np.asarray([vector], dtype=np.float32))
        self.ids[position] = candidate_id

    def delete(self, candidate_id: str) -> None:
        if candidate_id not in self.ids:
            return
        position = self.ids.index(candidate_id)
        self.index.remove_ids(np.array([position], dtype=np.int64))
        self.ids.pop(position)
