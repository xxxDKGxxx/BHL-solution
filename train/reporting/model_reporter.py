import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split

# Zakładam, że ten import masz u siebie
from train.reporting.model_interface import ModelInterface


class ModelReporter:
	def __init__(self, model_wrapper: ModelInterface, X, y, base_output_dir="reports"):
		if not isinstance(model_wrapper, ModelInterface):
			raise TypeError("Model musi implementować ModelInterface!")

		self.wrapper = model_wrapper
		self.X = X
		self.y = y
		self.base_output_dir = base_output_dir
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, stratify=y
		)
		self.current_report_dir = None
		self.report_txt_path = None

	def _setup_directories(self):
		"""Tworzy katalog dla bieżącego raportu z datą i godziną."""
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		model_name = self.wrapper.__class__.__name__
		folder_name = f"{timestamp}_{model_name}"

		self.current_report_dir = os.path.join(self.base_output_dir, folder_name)
		os.makedirs(self.current_report_dir, exist_ok=True)
		self.report_txt_path = os.path.join(self.current_report_dir, "summary.txt")

		print(f"[INFO] Raport zostanie zapisany w: {self.current_report_dir}")

	def _log(self, message):
		"""Wypisuje wiadomość na ekran ORAZ dopisuje do pliku raportu."""
		print(message)
		if self.report_txt_path:
			with open(self.report_txt_path, "a", encoding="utf-8") as f:
				f.write(message + "\n")

	def run_training(self):
		self._log("--- Rozpoczynanie treningu ---")
		self.wrapper.fit(self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test)
		self._log("Trening zakończony.")

	def plot_loss_history(self):
		history = self.wrapper.get_loss_history()

		if not history or not history.get('train_loss'):
			self._log("[INFO] Brak historii funkcji straty (Loss History). Pomijam wykres.")
			return

		plt.figure(figsize=(10, 5))
		plt.plot(history['train_loss'], label='Train Loss')

		if 'val_loss' in history and history['val_loss']:
			plt.plot(history['val_loss'], label='Validation Loss')

		plt.title('Funkcja Straty (Loss)')
		plt.xlabel('Iteracja / Epoka')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True)

		# Zapis do pliku
		save_path = os.path.join(self.current_report_dir, "loss_history.png")
		plt.savefig(save_path)
		plt.close()  # Zamykamy wykres, żeby zwolnić pamięć
		self._log(f"Zapisano wykres Loss: {save_path}")

	def plot_confusion_matrix(self):
		y_pred = self.wrapper.predict(self.X_test)
		cm = confusion_matrix(self.y_test, y_pred)

		plt.figure(figsize=(8, 6))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
		plt.title('Confusion Matrix')
		plt.ylabel('Prawdziwa klasa')
		plt.xlabel('Przewidziana klasa')

		# Zapis do pliku
		save_path = os.path.join(self.current_report_dir, "confusion_matrix.png")
		plt.savefig(save_path)
		plt.close()
		self._log(f"Zapisano macierz pomyłek: {save_path}")

	def run_cross_validation(self, k=5):
		self._log(f"\n--- Uruchamianie {k}-krotnej walidacji krzyżowej ---")

		kf = KFold(n_splits=k, shuffle=True, random_state=42)
		scores = []

		for fold_idx, (train_index, val_index) in enumerate(kf.split(self.X)):
			fold_model = self.wrapper.get_new_instance()
			X_fold_train, X_fold_val = self.X[train_index], self.X[val_index]
			y_fold_train, y_fold_val = self.y[train_index], self.y[val_index]

			fold_model.fit(X_fold_train, y_fold_train)

			preds = fold_model.predict(X_fold_val)
			acc = np.mean(preds == y_fold_val)
			scores.append(acc)

		self._log(f"Fold {fold_idx+1}: {acc:.4f}")

		self._log(f"Wyniki CV: {scores}")
		mean_score = np.mean(scores)
		std_score = np.std(scores)
		self._log(f"Średnia dokładność: {mean_score:.4f} (+/- {std_score:.4f})")

	def save_feature_importance(self):
		"""Oddzielna metoda do logowania ważności cech."""
		imps = self.wrapper.get_feature_importance()

		if imps:
			sorted_imps = sorted(imps.items(), key=lambda item: item[1], reverse=True)[:10]

			self._log("\n--- Top 10 Ważność cech ---")
			for k, v in sorted_imps:
				self._log(f"{k}: {v:.4f}")
		else:
			self._log("\n[INFO] Model nie zwraca ważności cech.")

	def generate_report(self):
		"""Główna metoda sterująca."""
		self._setup_directories()
		self._log(f"Raport wygenerowany: {datetime.now()}")
		self._log(f"Model Wrapper: {self.wrapper.__class__.__name__}")
		self._log("-" * 30)
		self.run_training()
		self.plot_loss_history()
		self.plot_confusion_matrix()
		self.run_cross_validation()
		self.save_feature_importance()

		print(f"\n[SUKCES] Cały raport zapisany w folderze: {self.current_report_dir}")