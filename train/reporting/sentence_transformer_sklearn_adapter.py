from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceTransformerSklearnAdapter(BaseEstimator, TransformerMixin):
	def __init__(self, model_name_or_path="all-mpnet-base-v2"):
		self.model_name_or_path = model_name_or_path
		self.model = SentenceTransformer(self.model_name_or_path)

	def transform(self, X, y=None):
		if hasattr(X, "tolist"):
			X = X.tolist()

		return self.model.encode(X)

	def fit(self, X, y=None):
		return self

	def get_feature_names_out(self, input_features=None):
		dim = self.model.get_sentence_embedding_dimension()
		return np.array([f"dim_{i}" for i in range(dim)])