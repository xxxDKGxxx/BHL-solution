import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime

from sklearn.manifold import TSNE
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
			if hasattr(vectorizer, 'vocabulary_'):
				f1_idx = vectorizer.vocabulary_[f1_name]
				f2_idx = vectorizer.vocabulary_[f2_name]
			else:
				# Obsługa SentenceTransformer (brak słownika słów, cechy to 'dim_X')
				# Zakładamy format "dim_123"
				f1_idx = int(f1_name.split('_')[-1])
				f2_idx = int(f2_name.split('_')[-1])
		except (KeyError, ValueError, IndexError):
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

	def plot_confusion_matrix_with_0_as_few_classes(self, external_dataset, main_class_name="Klasa główna"):
		y_pred = self.wrapper.predict(self.X_test)
		y_true_main = self.y_test
		
		test_df = pd.DataFrame({'question': self.X_test})
		merged_df = pd.merge(test_df, external_dataset, on='question', how='left').fillna(0)

		rest_categories = [col for col in external_dataset.columns if col != 'question']
		all_categories_ordered = [main_class_name] + rest_categories
		
		matrix_columns = []

		# Column for Main Class
		count_pred0_main = np.sum((y_pred == 0) & (y_true_main == 1))
		count_pred1_main = np.sum((y_pred == 1) & (y_true_main == 1))
		matrix_columns.append([count_pred0_main, count_pred1_main])

		# Columns for Rest Categories
		for cls in rest_categories:
			y_true_cls = merged_df[cls].values
			count_pred0 = np.sum((y_pred == 0) & (y_true_cls == 1))
			count_pred1 = np.sum((y_pred == 1) & (y_true_cls == 1))
			matrix_columns.append([count_pred0, count_pred1])
		
		if not matrix_columns:
			self._log("Nie można wygenerować macierzy 2xN - brak danych.")
			return
			
		matrix_data = np.array(matrix_columns).T
			
		plt.figure(figsize=(max(10, len(all_categories_ordered) * 1.2), 6))
		sns.heatmap(
			matrix_data,
			annot=True,
			fmt='d',
			cmap='viridis',
			xticklabels=all_categories_ordered,
			yticklabels=['Predykcja: 0', 'Predykcja: 1']
		)
		plt.title("Korelacja predykcji modelu z prawdziwą przynależnością do klas")
		plt.ylabel("Predykcja modelu dla klasy głównej")
		plt.xlabel("Prawdziwa przynależność próbki do klasy")
		plt.tight_layout()

		save_path = os.path.join(self.current_report_dir, "prediction_vs_true_class_correlation.png")
		plt.savefig(save_path)
		plt.close()
		self._log(f"Zapisano wykres korelacji predykcji z klasami: {save_path}")


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

	def plot_tsne(self):
		print("Generowanie mapy t-SNE (to może chwilę potrwać)...")

		# 1. Pobieramy wektorową reprezentację danych (TF-IDF/BoW)
		vectorizer = self.wrapper.get_vectorizer()
		X_vec = vectorizer.transform(self.X_test)
		
		if hasattr(X_vec, "toarray"):
			X_vec = X_vec.toarray()

		# 2. Redukcja do 2 wymiarów
		# perplexity=30 to standard, n_iter=1000 dla stabilności
		tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
		X_embedded = tsne.fit_transform(X_vec)

		# Tworzymy DataFrame dla łatwiejszego rysowania w Seaborn
		df_tsne = pd.DataFrame(X_embedded, columns=['x', 'y'])
		df_tsne['Kategoria'] = np.array(self.y_test)

		plt.figure(figsize=(10, 8))
		sns.scatterplot(
			data=df_tsne, x='x', y='y', hue='Kategoria',
			palette='viridis', s=60, alpha=0.7
		)
		plt.title("Mapa t-SNE: Jak bardzo podobne są do siebie pytania?")
		plt.show()  # lub savefig

	def plot_wordclouds(self):
		from wordcloud import WordCloud

		# Pobieramy unikalne kategorie
		categories = np.unique(self.y)

		plt.figure(figsize=(15, 5 * len(categories)))

		for i, cat in enumerate(categories):
			# Wyciągamy tekst tylko dla tej kategorii
			# (Zakładając, że self.X to surowe teksty)
			subset_idxs = (self.y == cat)
			text_subset = " ".join(self.X[subset_idxs])

			wc = WordCloud(width=800, height=400, background_color='white').generate(text_subset)

			plt.subplot(len(categories), 1, i + 1)
			plt.imshow(wc, interpolation='bilinear')
			plt.axis("off")
			plt.title(f"Słowa kluczowe dla: {cat}")

		plt.show()

	def plot_confidence_distribution(self):
			probs = self.wrapper.predict_proba(self.X_test)
			max_probs = np.max(probs, axis=1)  # Bierzemy pewność wygranej klasy

			# Dzielimy na poprawne i błędne predykcje
			preds = self.wrapper.predict(self.X_test)
			correct_mask = (preds == self.y_test)

			plt.figure(figsize=(10, 6))
			sns.histplot(max_probs[correct_mask], color='green', label='Poprawne', kde=True, bins=20, alpha=0.5)
			sns.histplot(max_probs[~correct_mask], color='red', label='Błędne', kde=True, bins=20, alpha=0.5)

			plt.xlabel("Pewność modelu (Prawdopodobieństwo)")
			plt.ylabel("Liczba przypadków")
			plt.title("Czy model jest pewny siebie, gdy się myli?")
			plt.legend()

			save_path = os.path.join(self.current_report_dir, "confidence_distr_plot.png")
			plt.savefig(save_path)

			self._log(f"Zapisano rozkład pewności modelu: {save_path}")

			plt.show()

	def save_model_and_datasets(self):
		with open(self.current_report_dir + "/model.pkl", 'wb') as f:
			pickle.dump(self.wrapper, f)

		save_test_df = pd.DataFrame()
		save_test_df["question"] = self.X_test
		save_test_df["target"] = self.y_test
		save_test_df.to_csv(self.current_report_dir + "/test_set.csv")

		save_train_df = pd.DataFrame()
		save_train_df["question"] = self.X_train
		save_train_df["target"] = self.y_train
		save_train_df.to_csv(self.current_report_dir + "/train_set.csv")

		self._log(f"Zapisano model i zbiór danych w {self.current_report_dir}/test_set.csv, "
		          f"{self.current_report_dir}/train_set.csv i "
		          f"{self.current_report_dir}/model.pkl")

	def plot_learning_curve(self, cv=5, n_points=5):
		"""
		Generuje wykres Learning Curve, aby sprawdzić wpływ wielkości zbioru danych na wynik.
		"""
		self._log("\nGenerowanie wykresu Learning Curve (to może potrwać)...")
		from sklearn.model_selection import learning_curve

		train_sizes = np.linspace(0.1, 1.0, n_points)

		train_sizes, train_scores, test_scores = learning_curve(
			self.wrapper, self.X, self.y,
			cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy'
		)

		train_mean = np.mean(train_scores, axis=1)
		train_std = np.std(train_scores, axis=1)
		test_mean = np.mean(test_scores, axis=1)
		test_std = np.std(test_scores, axis=1)

		plt.figure(figsize=(10, 6))

		plt.plot(train_sizes, train_mean, 'o-', color="#e74c3c", label="Wynik treningowy")
		plt.plot(train_sizes, test_mean, 'o-', color="#2ecc71", label="Wynik walidacji (CV)")

		plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#e74c3c")
		plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#2ecc71")

		plt.title(f"Learning Curve: {self.wrapper.__class__.__name__}")
		plt.xlabel("Liczba próbek treningowych")
		plt.ylabel("Dokładność (Accuracy)")
		plt.legend(loc="lower right")
		plt.grid(True, linestyle='--', alpha=0.7)

		if self.current_report_dir:
			save_path = os.path.join(self.current_report_dir, "learning_curve.png")
			plt.savefig(save_path)
			plt.close()
			self._log(f"Zapisano wykres Learning Curve: {save_path}")
		else:
			plt.show()

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
		# self.plot_learning_curve()
		self.save_model_and_datasets()


		print(f"\n[SUKCES] Cały raport zapisany w folderze: {self.current_report_dir}")