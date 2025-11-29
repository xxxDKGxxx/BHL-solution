import numpy as np

from sentence_transformers import SentenceTransformer
from interface.abstractembedder import Embedder


class SentenceTransformerEmbedder(Embedder):
	def embedd(self, v: str) -> np.ndarray[np.float32]:
		model = SentenceTransformer("all-mpnet-base-v2")
		embeddings = model.encode(v)

		return embeddings
