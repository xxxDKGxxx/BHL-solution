# Instrukcja wywoływania skryptów projektu

## Instalacja wymaganych modułów
Do uruchamiania skryptów można utworzyć wirtualne środowisko
```shell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Wymagane moduły zapisane są w pliku \app\requirements.txt. Należy je wszystkie zainstalować.

```shell
pip install -r .\app\requirements.txt
```

## Klasyfikacja kategorii pytań
W folderze \train znajdują się 4 podfoldery zawierające pliki z klasami wykorzystywanymi w skryptach oraz Jupyter Notebooks ze skryptami
do preprocessing'u danych, trenowania oraz testowania modeli. 

### Importowanie zbiorów danych i ich preprocessing
Ponieważ wykorzystywane przez nas zbiory danych są bardzo duże, załączamy osobno skompresowany foldery zawierający te zbiory
razem z odpowiednią strukturą plików do wywoływanych później skryptów. Po pobraniu go trzeba go wstawić do folderu 
\train\datasets_preprocessing.  
Po przeniesieniu zbiorów danych do wskazanego miejsca można wywołać skrypty z pliku \train\datasets_preprocessing\datasets_preprocessing.ipynb,
które przepiszą je do dalej używanych plików .csv, które zostaną zapisane w folderze \train\datasets_preprocessing\csv_question_files.

### Trenowanie modeli
W plikach Jupyter Notebooks \train\one_model_training\svm_training_one-model.ipynb oraz 
\train\three_models_training\train_three_models.ipynb znajdują się skrypty wywołujące procesy trenowania stworzonych przez nas modeli,
generowania raportów i wizualizacji danych. Pliki raportów zapisywane są w folderach \reports tworzonych wewnątrz folderów,
w których znajdują się wskazane pliki.

### Generowanie zbiorów testowych do porównywania modeli
Generowane raporty podczas trenowania zawierają pliki .csv ze zbiorami treningowym i testowym. Aby porównanie działania 
obydwu modeli było rzetelne, stworzony został skrypt do generowania zbiorów testowych z pytaniami, które nie powtórzyły się 
w żadnym z modeli podczas procesu ich trenowania. Modele do porównywania wraz z ich zbiorami treningowymi train_set.csv zostały zapisane
w folderze \train\saved_models. Opisany skrypt odczytuje pliki zbiorów treningowych i porównuje 
je z pełnymi zbiorami danych, by następnie wygenerować trzy zbiory testowe z innymi pytaniami.
Zapisane one zostają w folderze \train\datasets_preprocessing\test_all_models.  
Skrypt do wywołania znajduje się w pliku \train\three_models_training\choose_test_set_for_multiple_models.ipynb.

### Testowanie modeli
Skrypty wywołujące procesy testowania modeli w celu ich porównania znajdują się w plikach \train\one_model_training\svm_test_one_model.ipynb 
oraz \train\three_models_training\test_three_models_together.ipynb. Korzystają one ze zbiorów testowych zapisanych w folderze \train\datasets_preprocessing\test_all_models.
Ładują one obiekty wytrenowanych modeli z plików .pkl zapisanych w folderze \train\saved_models. Aby obiekty modeli zostały poprawnie
załadowane, przed wywołaniem skryptu czytającego pliki .pkl konieczne jest zaimportowanie klas tych modeli:  
- dla 1 modelu
```python
from train.reporting.model_interface import ModelInterface
from train.reporting.svm_model_wrapper import SVMModelWrapper
```
- dla 3 modeli
```python
from train.reporting.model_interface import ModelInterface
from train.reporting.text_svm_wrapper import TextSVMWrapper 
```

