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

	def plot_top_2_features_boundary(self):
		"""
		Uniwersalna metoda:
		1. Pyta wrapper o najważniejsze cechy.
		2. Klonuje klasyfikator (niezależnie czy to SVM, RF czy inny).
		3. Trenuje klona na 2 cechach i rysuje granice.
		"""
		print("Generowanie wykresu 2D dla topowych cech (wersja uniwersalna)...")

		# 1. Pobieramy ważność cech z interfejsu (działa dla każdego modelu, który to obsługuje)
		imps = self.wrapper.get_feature_importance()
		if not imps:
			print("[INFO] Model nie zwrócił ważności cech. Pomijam wykres 2D.")
			return

		# Sortujemy i bierzemy 2 najlepsze słowa
		sorted_imps = sorted(imps.items(), key=lambda item: item[1], reverse=True)
		if len(sorted_imps) < 2:
			print("[INFO] Za mało cech, by narysować wykres 2D.")
			return

		f1_name = sorted_imps[0][0]
		f2_name = sorted_imps[1][0]
		print(f"Wybrano cechy do wykresu: '{f1_name}' oraz '{f2_name}'")

		# 2. Szukamy wektoryzatora i klasyfikatora w pipeline
		pipeline = self.wrapper.model
		vectorizer = None
		classifier = None

		# Przeszukujemy kroki pipeline'u
		if hasattr(pipeline, 'named_steps'):
			# Szukamy wektoryzatora (ma metodę transform i get_feature_names_out)
			for name, step in pipeline.named_steps.items():
				if hasattr(step, 'transform') and hasattr(step, 'get_feature_names_out'):
					vectorizer = step
				# Zakładamy, że ostatni krok to klasyfikator
				if step == pipeline.steps[-1][1]:
					classifier = step
		else:
			print("[BLAD] Model we wrapperze nie jest Pipeline'em sklearn.")
			return

		if vectorizer is None:
			print("[BLAD] Nie znaleziono wektoryzatora w pipeline.")
			return

		# 3. Znajdujemy indeksy tych słów w macierzy
		# Wektoryzatory mają słownik vocabulary_ {słowo: index}
		try:
			f1_idx = vectorizer.vocabulary_[f1_name]
			f2_idx = vectorizer.vocabulary_[f2_name]
		except KeyError:
			print("[BLAD] Nie udało się znaleźć indeksów dla wybranych słów.")
			return

		# 4. Przygotowujemy dane (tylko 2 kolumny)
		# Transformujemy X_test (tekst) -> Macierz
		X_full = vectorizer.transform(self.X_test)

		# Konwersja do dense array (wymagane do slicingu i wykresów)
		if hasattr(X_full, "toarray"):
			X_full = X_full.toarray()

		# Wycinamy tylko interesujące nas kolumny: [wszystkie wiersze, [kol1, kol2]]
		X_2d = X_full[:, [f1_idx, f2_idx]]

		# Kodowanie etykiet na liczby (dla kolorów na wykresie)
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		y_encoded = le.fit_transform(self.y_test)

		# 5. Klonujemy klasyfikator (to jest klucz do uniwersalności!)
		from sklearn.base import clone
		clf_2d = clone(classifier)  # Tworzy czystą kopię np. SVM, RandomForest, itd.

		# Trenujemy "mały" model na 2 cechach
		try:
			clf_2d.fit(X_2d, y_encoded)
		except Exception as e:
			print(f"[INFO] Nie udało się wytrenować modelu 2D (może model nie obsługuje sparse?): {e}")
			return

		# 6. Rysowanie (Meshgrid)
		import matplotlib.pyplot as plt
		import numpy as np

		plt.figure(figsize=(10, 8))

		# Marginesy
		x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
		y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

		# Dodajemy margines 10%
		x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
		y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1

		xx, yy = np.meshgrid(
			np.linspace(x_min - x_margin, x_max + x_margin, 100),
			np.linspace(y_min - y_margin, y_max + y_margin, 100)
		)

		# Predykcja tła
		# Spłaszczamy siatkę, robimy predykcję i wracamy do kształtu siatki
		Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

		X_plot_x = X_2d[:, 0]
		X_plot_y = X_2d[:, 1]

		# Rysujemy punkty
		scatter = plt.scatter(X_plot_x, X_plot_y, c=y_encoded,
		                      s=50, edgecolors='k', cmap='coolwarm', alpha=0.7)

		plt.xlabel(f"Cecha: '{f1_name}'")
		plt.ylabel(f"Cecha: '{f2_name}'")
		plt.title(f"Granice decyzyjne ({classifier.__class__.__name__})")

		# Legenda
		try:
			plt.legend(handles=scatter.legend_elements()[0], labels=le.classes_.tolist(), title="Klasy")
		except:
			pass  # Czasami legend_elements rzuca błąd przy specyficznych danych

		# Zapis
		if self.current_report_dir:
			import os
			save_path = os.path.join(self.current_report_dir, "decision_boundary_2d.png")
			plt.savefig(save_path)
			plt.close()
			self._log(f"Zapisano wykres 2D: {save_path}")
		else:
			plt.show()

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
				val_as_float = float(v) if np.ndim(v) == 0 or np.size(v) == 1 else v
				self._log(f"{k}: {val_as_float:.4f}")
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
		self.plot_top_2_features_boundary()

		print(f"\n[SUKCES] Cały raport zapisany w folderze: {self.current_report_dir}")