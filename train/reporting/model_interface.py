from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Self
import numpy as np


class ModelInterface(ABC):
	"""
	Abstrakcyjna klasa bazowa definiująca interfejs,
	którego oczekuje klasa raportująca.
	"""

	@abstractmethod
	def fit(self, X_train: np.ndarray, y_train: np.ndarray,
	        X_val: Optional[np.ndarray] = None,
	        y_val: Optional[np.ndarray] = None) -> None:
		"""
		Trenuje model.
		Opcjonalnie przyjmuje zbiór walidacyjny do śledzenia historii uczenia (loss curve).
		"""
		pass

	@abstractmethod
	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Zwraca predykcje klas (np. [0, 1, 0])."""
		pass

	@abstractmethod
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""Zwraca prawdopodobieństwa (dla ROC/AUC)."""
		pass

	@abstractmethod
	def get_loss_history(self) -> Dict[str, List[float]]:
		"""
		Zwraca słownik z historią funkcji straty, np.:
		{
			'train_loss': [0.5, 0.4, 0.1],
			'val_loss': [0.6, 0.5, 0.2]
		}
		Jeśli model nie wspiera historii iteracji (np. standardowy KNN), zwraca pusty słownik.
		"""
		pass

	@abstractmethod
	def get_feature_importance(self) -> Optional[Dict[str, float]]:
		"""Zwraca ważność cech, jeśli model to wspiera."""
		pass

	# @abstractmethod
	# def get_params(self) -> Dict[str, Any]:
	# 	"""Zwraca hiperparametry modelu (do logowania w raporcie)."""
	# 	pass

	@abstractmethod
	def get_new_instance(self) -> Self :
		pass

	@abstractmethod
	def get_vectorizer(self):
		pass