import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split

from train.reporting.model_interface import ModelInterface


class ModelReporter:
	def __init__(self, model_wrapper: ModelInterface, X, y):
		if not isinstance(model_wrapper, ModelInterface):
			raise TypeError("Model musi implementować ModelInterface!")

		self.wrapper = model_wrapper
		self.X = X
		self.y = y
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, stratify=y
		)

	def run_training(self):
		"""Uruchamia trening poprzez wrapper."""
		print("Rozpoczynanie treningu...")
		# Przekazujemy dane walidacyjne, żeby wrapper mógł (jeśli umie) zebrać historię
		self.wrapper.fit(self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test)
		print("Trening zakończony.")

	def plot_loss_history(self):
		"""Rysuje wykres funkcji straty, jeśli wrapper dostarczył dane."""
		history = self.wrapper.get_loss_history()

		if not history or not history.get('train_loss'):
			print("[INFO] Ten model nie udostępnia historii funkcji straty (Loss History). Pomijam wykres.")
			return

		plt.figure(figsize=(10, 5))
		plt.plot(history['train_loss'], label='Train Loss')

		if 'val_loss' in history and history['val_loss']:
			plt.plot(history['val_loss'], label='Validation Loss')

		plt.title('Funkcja Straty (Loss) w kolejnych iteracjach')
		plt.xlabel('Iteracja / Epoka')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True)
		plt.show()

	def plot_confusion_matrix(self):
		y_pred = self.wrapper.predict(self.X_test)
		cm = confusion_matrix(self.y_test, y_pred)

		plt.figure(figsize=(8, 6))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
		plt.title('Confusion Matrix')
		plt.ylabel('Prawdziwa klasa')
		plt.xlabel('Przewidziana klasa')
		plt.show()

	def run_cross_validation(self, k=5):
		"""
		Ręczna implementacja CV, używając metody get_new_instance() z wrappera.
		Uniezależnia nas to od funkcji cross_val_score ze sklearn, która wymagałaby
		obiektu sklearn, a my mamy nasz wrapper.
		"""
		print(f"Uruchamianie {k}-krotnej walidacji krzyżowej...")

		kf = KFold(n_splits=k, shuffle=True, random_state=42)
		scores = []

		for train_index, val_index in kf.split(self.X):
			fold_model = self.wrapper.get_new_instance()
			X_fold_train, X_fold_val = self.X[train_index], self.X[val_index]
			y_fold_train, y_fold_val = self.y[train_index], self.y[val_index]

			fold_model.fit(X_fold_train, y_fold_train)

			preds = fold_model.predict(X_fold_val)
			acc = np.mean(preds == y_fold_val)

			scores.append(acc)

		print(f"Wyniki CV: {scores}")
		print(f"Średnia dokładność: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

	def generate_report(self):
		self.run_training()
		self.plot_loss_history()
		self.plot_confusion_matrix()
		self.run_cross_validation()

		imps = self.wrapper.get_feature_importance()

		if imps:
			sorted_imps = sorted(imps.items(), key=lambda item: item[1], reverse=True)[:10]

			print("\nTop 10 Ważność cech:")

			for k, v in sorted_imps:
				print(f"{k}: {v:.4f}")