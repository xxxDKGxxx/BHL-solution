from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any
from train.reporting.model_interface import ModelInterface


class TextSVMWrapper(ModelInterface):
	def get_params(self) -> Dict[str, Any]:
		return self.model.get_params()

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict_proba(X)

	def get_vectorizer(self):
		return self.model.steps[0][1]

	def __init__(self, C=1.0):
		self.model = Pipeline([
			('tfidf', TfidfVectorizer(stop_words='english')),
			('svm', SVC(C=C, random_state=42, probability=True, kernel='linear'))
		])
		self.C = C
		self.is_fitted = False

	def fit(self, X, y, X_val=None, y_val=None):
		self.model.fit(X, y)
		self.is_fitted = True

	def predict(self, X):
		return self.model.predict(X)

	def get_loss_history(self):
		return {}

	def get_new_instance(self):
		return TextSVMWrapper(C=self.C)

	def get_feature_importance(self):
		if not self.is_fitted:
			return {}

		try:
			feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
			coefs = self.model.named_steps['svm'].coef_.copy()

			avg_coefs = np.mean(np.abs(coefs), axis=0)
			avg_coefs = np.ravel(avg_coefs)

			# Tworzymy słownik {słowo: waga}
			importance_dict = dict(zip(feature_names, avg_coefs))
			return importance_dict
		except Exception as e:
			print(f"Nie udało się pobrać ważności cech: {e}")
			return {}