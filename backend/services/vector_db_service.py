import faiss
import numpy as np
from typing import Tuple


class VectorDBService:
    def __init__(self):
        self.index = None
        self.documents = []

    def store(self, texts: list, vectors: np.ndarray) -> None:
        """
        Populates the vector index and stores associated text documents.

        Args:
            texts (List[str]): A list of raw text strings to be stored.
            vectors (np.ndarray): A NumPy array of embeddings corresponding to the texts.

        Returns:
            None
        """
        self.documents = texts
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 1) -> Tuple[str, float]:
        """
        Performs a similarity search within the FAISS index.

        Args:
            query_vector (np.ndarray): The vectorized representation of the user's query.
            top_k (int): The number of similar documents to retrieve. Defaults to 1.

        Returns:
            Tuple[str, float]: A tuple containing the matched text and a confidence
                score (typically normalized between 0 and 1).
        """
        distances, indices = self.index.search(query_vector, top_k)
        score = 1 / (1 + distances[0][0])
        return self.documents[indices[0][0]], score
