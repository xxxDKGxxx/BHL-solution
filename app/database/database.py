from typing import override
import faiss
import numpy as np
from app.interface.abstractdatabasecontext import DatabaseContext


class Database(DatabaseContext):
    def __init__(self, embedder):
        """
        embedder – obiekt posiadający metodę .embed(text: str) -> list[float]
        """
        self.embedder = embedder
        self.documents = []  # lista: (question, answer)
        self.index = None
        self.dim = None

    @override
    def insert(self, prompt: str, answer: str) -> bool:
        """
        Wstawia nowy dokument do FAISS.
        Format dokumentu:
            "Question: <prompt>\nAnswer: <answer>"
        """
        #doc_text = f"Prompt: {prompt}\nAnswer: {answer}"
        doc_text = prompt
        embedding = np.array(self.embedder.embedd(doc_text), dtype="float32")

        # Inicjalizacja indeksu FAISS jeśli jeszcze nie istnieje
        if self.index is None:
            self.dim = embedding.shape[0]
            self.index = faiss.IndexFlatL2(self.dim)

        # Dodanie wektora
        self.index.add(np.array([embedding]))
        self.documents.append((prompt, answer))
        return True

    @override
    def get(self, prompt: str) -> tuple[str, float]:
        """
        Zwraca odpowiedź i dystans najbliższego wektora.
        """
        if self.index is None or len(self.documents) == 0:
            return "", 0.0

        query_vec = np.array(
            self.embedder.embedd(prompt),
            dtype="float32"
        ).reshape(1, -1)

        distances, indices = self.index.search(query_vec, 1)
        idx = indices[0][0]
        dist = float(distances[0][0])

        # pobranie odpowiedzi
        _, answer = self.documents[idx]
        return answer, dist