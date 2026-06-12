# Projekt NLP, Wprowadzenie do Sztucznej Inteligencji

## Andrzej Wrzesiński, Dominik Zieliński

### 1. Wprowadzenie

Celem projektu było rozszerzenie funkcjonalności systemu zbudowanego w ramach hackathonu Best Hacking League odbytego w zeszłym roku. Zamierzyliśmy dotrenować osobne modele tematyczne SVM do klasyfikacji promptów z trzech dziedzin: biologii, matematyki i programowania na bardzo dużych zbiorach danych. Dodatkowo wytrenowaliśmy klasyfikator SVM do oceny, czy prompt jest generatywny (odpowiedź bardziej złożona i kreatywna) lub nie (prosta odpowiedź o fakt bądź definicję). Wszystkie wytrenowane modele postanowiliśmy zintegrować z systemem, by porównać zysk czasowy i pamięciowy przed i po integracji.

### 2. Opis metod i wykorzystanych zbiorów danych

#### Zbiory danych i metody dla klasyfikatorów tematycznych

Ta część realizowanego projektu opierała się na wytrenowaniu i zintegrowaniu z systemem zespołu modeli klasyfikujących prompty do określonych obszarów tematycznych. Wykorzystany zbiór to ogromna baza tytułów i opisów postów na StackExchange z trzech dziedzin: biologicznej, matematycznej oraz programistycznej w postaci response'ów JSON. Preprocessing danych opierał się na parsowaniu JSON-a i wyciągnięcie samych tytułów. Ze względu na to, iż każdy obszar tematyczny to był osobny zbiór danych musieliśmy jeszcze je ze sobą złączyć i odpowiedniu wyróżnić przy łączeniu za pomocą trzech kolumn binarnych (każda dla innego obszaru tematycznego). Zdecydowaliśmy się na taką organizację danych ze względu na architekturę rozwiązania - mamy trzy modele które potrafią odróżnić jedną kategorię od dwóch innych, dlatego były potrzebne trzy kolumny z targetem. Ostatecznie do kodowania promptów wykorzystaliśmy TF-IDF z biblioteki Scikit-Learn. Wybraliśmy standardowe ```stop_words='english'```, aby odfiltrować nic nie znaczące słowa i zapobiec przeuczeniu. Zadbaliśmy o zbalansowanie klas budując dla każdego modelu osobno dataset w proporcjach klas 2\:1\:1, 2 to klasa przewidywana przez model a dwie jedynki to dwie inne, żeby propocja 0 i 1 była 50%:50%.
Część preprocessingu mieliśmy już przygotowaną, należało jedynie dorzucić więcej danych i wytrenować trzy klasyfikatory. Użyliśmy Maszyn Wektorów Nośnych z jądrem liniowym i parametrem kosztu wynoszącym ```1.0```. Po treningu zajęliśmy się ich integracją z systemem. W tym celu dodaliśmy klasę ```abstracttopicclassifier``` oraz jej implementację w postaci ```multi_model_topic_classifier```. Predykcja polega na przepuszczeniu prompt'u przez każdy model, wyciągnięcia prawdopodobieństwa przynależności do swojej klasy, wybranie modelu które zwrócił największe prawdopodobieństwo i ostatecznie jeśli jest większe niż ustalony threshold (wybrany 0.9) zwracana jest nazwa klasy w postaci ciągu znaków (w przypadku wyniku mniejszego niż threshold zwracany jest wynik "General").

#### Zbiór danych i metody dla klasyfikatora Fact or Generative

Ponieważ nie znaleźliśmy idealnego zbioru danych wraz z kategoriami, czy prompt został wpisany w celu udzielenia odpowiedzi generatywnej bądź nie, wykorzystaliśmy w tym celu zbiór z pytaniami najbardziej zbliżony do naszych potrzeb. Zawierał on przede wszystkim listę ok. 15 tysięcy promptów, ale także kategorii, do których one należą, jednak w tym zbiorze były one bardziej szczegółowe. Postanowiliśmy więc zmodyfikować go, poddając go dokładnej analizie. Sprawdzaliśmy semantycznie, dla jakich kategorii prompty są przeważnie generatywne, a kiedy nie, a także szukaliśmy najbardziej podstawowych wzorców występujących najczęściej w pytaniach obydwu kategorii, które by jak najlepiej określały, jakiego typu jest dane pytanie.  
W celu modyfikacji wykorzystywanego datasetu utworzyliśmy specjalny pipeline do preprocessingu danych. Oprócz przekształcania wierszy, całość przepisywana była również z pliku .jsonl do formatu .csv. Ocenialiśmy różne metody określania typów promptów, ostatecznie najdokładniej udało nam się przypisać je, dzięki utworzeniu kluczy w postaci prostych wzorców regularnych regex oraz określenia kategorii z pierwotnego datasetu, których znaczna większość pytań należy do tego samego typu. Wzorce regularne
miały większą wagę przy określaniu kategorii prompta, w dalszej kolejności był sprawdzany rozkład kategorii pierwotnych. Pytania niepewne były pomijane w zbiorze docelowym. Dla najdokładniejszego zbioru dokładnie 11720 promptów było przypisanych do pytań faktowych, a 2207 do generatywnych.  
Podobnie jak dla klasyfikatorów tematycznych, do kodowania promptów wykorzystaliśmy TF-IDF, jednakże, zamiast ```stop_words='english``` użyliśmy, ```stop_words=None``` - jak się okazało, przy tym wyborze trening wychodzi zauważalnie lepiej. Do treningu użyliśmy również modelu SVM z kosztem 1.0. Zależało nam też na odpowiednich proporcjach kategorii - wyrównaliśmy je i w używanym do treningu datasecie znajduje się po 2000 pytań z obydwu kategorii.

#### Opis używanych technologii

Do tworzenia modeli wykorzystaliśmy najpopularniejsze biblioteki Python. Przede wszystkim skupiliśmy się na bibliotece scikit-learn (głównie TfidfVectorizer i SVM). Użyliśmy również, między innymi do preprocessingu danych i obsługi datasetów biblioteki pandas i numpy. Do pomocy w odczycie i odpowiednim zapisie oraz modyfikacji zbiorów danych użyliśmy bibliotek json oraz csv, które działają z plikami o tych formatach. Wykorzystaliśmy też moduł pickle do zapisu wytrenowanych modeli. Odpowiednie wywołanie funkcji do preprocessingu i treningów modeli zapisaliśmy w notatniku Jupyter.

### 3. Wyniki

#### Czasy trenowania klasyfikatorów tematycznych
Przy okazji trenowania klasyfikatorów tematycznych zbadaliśmy na modelu matematycznym czasy trenowania w zależności od rozmiaru danych (z dokładnością do 1s).

| Rozmiar | Czas |
| :--- | :--- |
| 12k | < 1s |
| 24k | 70s |
| 48k | 323s |
| 96k | 1258s |
| max | 1477s |

Ze względu na to iż czas trenowania dla całego możliwego zbioru danych był dla nas akceptowalny to wytrenowaliśmy wszystkie modele na całym możliwym zbiorze danych (ze względu na balansowanie klas ograniczeniem dla nas była liczba danych kategorii biologicznej - było ich najmniej, a każdy model do trenowania dostawał 50% danych swojej klasy i po 25% dwóch innych).

#### Wpływ dodania większej liczby danych na wyniki na zbiorze testowym
Porównaliśmy jak większa liczba danych wpłynęła na wyniki na zbiorze testowym.

##### Model matematyczny (programistyczny podobnie)

Wyniki bazowe

```
Acc=0.9021, Prec=0.9021, Rec=0.9018, F1=0.9020
```

Wyniki po wytrenowaniu na większej liczbie rekordów

```
Accuracy:  0.9299
Precision: 0.9300 (macro)
Recall:    0.9299 (macro)
F1-score:  0.9299 (macro)
```

##### Model biologiczny
Ciekawe wyniki uzyskaliśmy dla modelu biologicznego, gdyż jego accuracy było nieco wyższe niż dla matematycznego i programistycznego modelu.

Wyniki bazowe
```
Acc=0.9413, Prec=0.9415, Rec=0.9412, F1=0.9412
```

Wyniki po wytrenowaniu na większej liczbie rekordów
```
Wyniki na zbiorze testowym
Accuracy:  0.9648
Precision: 0.9648 (macro)
Recall:    0.9648 (macro)
F1-score:  0.9648 (macro)
```


Jak widać analizowaliśmy cztery metryki i na zbiorze testowym uzyskaliśmy niemal identyczne ich wartości. W każdym przypadku większa liczba danych poprawiła wynik o 2-3%.


#### Analiza pomyłek modeli tematycznych

Przeanalizowaliśmy macierz pomyłek dla każdego modelu na trzech klasach aby zbadać, z którymi tematami modele mają największy problem. Okazało się, iż model matematyczny najbardziej się myli przy przewidywaniu pytań z dziedziny programowania (i vice versa - model programowania myli się na pytaniach matematycznych).

Model biologiczny nie miał problemów z odróżnianiem swoich pytań od matematycznych czy programistycznych.


#### Trening klasyfikatora Fact or Generative

Dla naszego treningu utworzyliśmy DataFrame z 2000 pytań generatywnych i niegeneratywnych. Treningi modelu ocenialiśmy dla różnych metryk, otrzymując wyniki dla treningu kroswalidacyjnego i następnie jego testu. Otrzymaliśmy bardzo wysokie wyniki dla każdej z tych metryk. Przykładowe wartości przedstawiamy poniżej:

- test_f1_macro: 0.9549 (+/- 0.0069)
- train_f1_macro: 0.9921 (+/- 0.0008)
- test_precision_macro: 0.9550 (+/- 0.0069)
- train_precision_macro: 0.9922 (+/- 0.0007)
- test_recall_macro: 0.9549 (+/- 0.0069)
- train_recall_macro: 0.9921 (+/- 0.0008)

Klasa FactClassifierReporter wzbogacona jest dodatkowo o funkcje tworzące wykresy t-SNE, wykresy wordclouds, confussion matrix do oceny ilości pomyłek oraz confidence distribution, na którym widać, że nie tylko model poprawnie przewiduje rodzaj prompta, ale robi to bardzo pewnie, tj. dla praktycznie wszystkich testowanych przypadków pewność modelu jest przynajmniej 90-procentowa.

Przy treningu skupiliśmy się przede wszystkim na wykorzystaniu vectorizer'a TF-IDF, jednak chcieliśmy też sprawdzić, jak model zadziała z innym vectorizer'em. Przetestowaliśmy więc jeszcze CountVectorizer'a. Jak się jednak okazało, wyniki niewiele się różniły od modelu działającego z TF-IDF.

#### Zysk czasowy i pamięciowy integracji modeli

Zgodnie z naszymi założeniami, dokonaliśmy integracji naszych wytrenowanych modeli z wcześniej zaimplementowanym systemem. Chcieliśmy ocenić, jak nasze modele działają w praktyce, ale też przede wszystkim, czy ta integracja jest optymalna pod względem szybkości czy zużycia pamięci. Wprowadziliśmy więc do systemu kilka promptów (tych samych dla systemu przed i po integracji) takich, które nie ulegną procesowi cachowania do bazy i porównaliśmy, jak zmienia się czas generowania odpowiedzi po tej integracji.  
Uśredniony czas dla systemu przed integracją wyniósł ok. 14.66 s, natomiast po ok. 16.15 s. Jak widać, czas generowania odpowiedzi przy użyciu naszych modeli SVM zwiększył się.

### 4. Wnioski

#### Czas trenowania modeli tematycznych

Dużym zaskoczeniem dla nas był czas trenowania modeli. Cały zbiór miał około 100 tys rekordów, a i tak czas trenowania zajął mniej więcej 20-30 minut. Dlatego zdecydowaliśmy się wytrenować każdy klasyfikator na maksymalnej możliwej liczbie danych. Maszyny wektorów nośnych zatem trenują się bardzo szybko, przynajmniej na jądrze liniowym.

#### Metryki i analiza pomyłek modeli tematycznych

Spodziewanym rezultatem było polepszenie wartości metryk na większej liczbie danych, jednak ciekawe były zawyżone wyniki modelu biologicznego. Wyjaśniła je dopiero analiza pomyłek - modele matematyczne i programistyczny miały problemy z wzajemnym odróżnieniem swoich pytań. Przeanalizowaliśmy pojedyncze próbki rekordów i wysnuliśmy dwa wnioski który potencjalnie wyjaśnia zjawisko:

1. Czasami ciężko jest odróżnić do jakiej dziedziny należy dane pytanie, gdyż przykładowo niektóre pytania algorytmiczne możnaby zaklasyfikować do jednej oraz do drugiej klasy jednocześnie. Być może należałoby przelabelować dataset tak, aby jedno pytanie mogło należeć do kilku kategorii jednocześnie.
2. Zdarzały się czasami mislabele, gdzie pytanie programistyczne było klasyfikowane jako matematyczne (lecz jest to też subiektywna opinia i były to pytania raczej na pograniczu dwóch dziedzin).
3. 
Cięzko byłoby jednak te dwie rzeczy wyeliminować, gdyż wymagałoby to ręcznej i subiektywnej korekcji i filtrowania zbioru danych.

Jakość dostępnych informacji treningowym ma zatem duży wpływ na wyniki modelu, w szczególności kiedy są one niejednoznacze na przestrzeni wielu klas.


#### Wnioski dla klasyfikatora promptów generatywnych
Wyniki testowe niewiele odbiegają od wartości treningowych, także możemy wykluczyć zjawisko overfittingu. Dodatkowo, dla wszystkich metryk model uzyskuje wysoką stabilność, o czym świadczą niskie wartości błędów. Wykresy wygenerowane przez klasę reportera dodatkowo pokazują, że model działa przy tym bardzo pewnie, czyli z wysokim prawdopodobieństwem przewiduje poprawne kategorie promptów.

#### Integracja modeli
Integracja modeli SVM nieznacznie zwiększyła czas działania generowania odpowiedzi. Zakładamy jednak, że jest to przede wszystkim spowodowane faktem, że modele SVM są ciężkie, a ich wykorzystanie jest kosztowne czasowo. Dodatkowo czas pogorszyć mogło rozbicie jednego modelu na trzy osobne modele dziedzinowe.

### 5. Linki

- Zbiór danych do klasyfikatora promptów generatywnych: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Zbiór danych do klasyfikatorów tematycznych: https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl
- Scikit-learn https://scikit-learn.org/stable/
- TF-IDF https://pl.wikipedia.org/wiki/TFIDF
- 