# Projekt NLP, Wprowadzenie do Sztucznej Inteligencji

## Andrzej Wrzesiński, Dominik Zieliński

### 1. Wprowadzenie
Celem projektu było rozszerzenie funkcjonalności systemu zbudowanego w ramach hackathonu 
Best Hacking League odbytego w zeszłym roku. Zamierzyliśmy dotrenować osobne modele tematyczne SVM 
do klasyfikacji promptów z trzech dziedzin: biologii, matematyki i programowania na bardzo dużych zbiorach danych.
Dodatkowo wytrenowaliśmy klasyfikator SVM do oceny, czy prompt jest generatywny (odpowiedź bardziej złożona i kreatywna)
lub nie (prosta odpowiedź o fakt bądź definicję). Wszystkie wytrenowane modele postanowiliśmy zintegrować z systemem, by porównać
zysk czasowy i pamięciowy przed i po integracji.

### 2. Opis metod i wykorzystanych zbiorów danych

#### 2.4. Zbiór danych dla klasyfikatora Fact or Generative
Ponieważ nie znaleźliśmy idealnego zbioru danych wraz z kategoriami, czy prompt został wpisany w celu udzielenia odpowiedzi
generatywnej bądź nie, wykorzystaliśmy w tym celu zbiór z pytaniami najbardziej zbliżony do naszych potrzeb. Zawierał on przede
wszystkim listę ok. 15 tysięcy promptów, ale także kategorii, do których one należą, jednak w tym zbiorze były one 
bardziej szczegółowe. Postanowiliśmy więc zmodyfikować go, poddając go dokładnej analizie. Sprawdzaliśmy semantycznie, 
dla jakich kategorii prompty są przeważnie generatywne, a kiedy nie, a także szukaliśmy najbardziej podstawowych wzorców
występujących najczęściej w pytaniach obydwu kategorii, które by jak najlepiej określały, jakiego typu jest dane pytanie.  
W celu modyfikacji wykorzystywanego datasetu utworzyliśmy specjalny pipeline do preprocessingu danych. Oprócz przekształcania
wierszy, całość przepisywana była również z pliku .jsonl do formatu .csv. Ocenialiśmy różne metody określania typów promptów,
ostatecznie najdokładniej udało nam się przypisać je, dzięki utworzeniu kluczy w postaci prostych wzorców regularnych regex
oraz określenia kategorii z pierwotnego datasetu, których znaczna większość pytań należy do tego samego typu. Wzorce regularne
miały większą wagę przy określaniu kategorii prompta, w dalszej kolejności był sprawdzany rozkład kategorii pierwotnych. Pytania
niepewne były pomijane w zbiorze docelowym. Dla najdokładniejszego zbioru dokładnie 11720 promptów było przypisanych do 
pytań faktowych, a 2207 do generatywnych. 

### 3. Wyniki

#### 3.2. Trening klasyfikatora Fact or Generative
Trening polegał na wywołaniu treningu z Cross Validation opartej na 5 foldach z trzema powtórzeniami.
Wstępnie należało jednak wyodrębnić poszczególne pytania, naszym celem było wyrównanie ilości pytań dla obydwu 
wartości targeta. Dla naszego treningu utworzyliśmy DataFrame z 2000 pytań generatywnych i niegeneratywnych.  
Treningi modelu ocenialiśmy dla różnych metryk, otrzymując wyniki dla treningu kroswalidacyjnego i następnie jego testu.
Otrzymaliśmy bardzo wysokie wyniki dla każdej z tych metryk. Przykładowe wartości przedstawiamy poniżej:
- test_f1_macro: 0.9549 (+/- 0.0069)
- train_f1_macro: 0.9921 (+/- 0.0008)
- test_precision_macro: 0.9550 (+/- 0.0069)
- train_precision_macro: 0.9922 (+/- 0.0007)
- test_recall_macro: 0.9549 (+/- 0.0069)
- train_recall_macro: 0.9921 (+/- 0.0008)

Wyniki testowe niewiele odbiegają od wartości treningowych, także możemy wykluczyć zjawisko overfittingu. Dodatkowo, dla 
wszystkich metryk model uzyskuje wysoką stabilność, o czym świadczą niskie wartości błędów.  
Klasa FactClassifierReporter wzbogacona jest dodatkowo o funkcje tworzące wykresy t-SNE, wykresy wordclouds, confussion matrix
do oceny ilości pomyłek oraz confidence distribution, na którym widać, że nie tylko model poprawnie przewiduje rodzaj prompta,
ale robi to bardzo pewnie, tj. dla praktycznie wszystkich testowanych przypadków pewność modelu jest przynajmniej 90-procentowa.

#### 3.3. Zysk czasowy i pamięciowy integracji modeli
Zgodnie z naszymi założeniami, dokonaliśmy integracji naszych wytrenowanych modeli z wcześniej zaimplementowanym systemem.
Chcieliśmy ocenić, jak nasze modele działają w praktyce, ale też przede wszystkim, czy ta integracja jest optymalna pod względem
szybkości czy zużycia pamięci. Wprowadziliśmy więc do systemu kilka promptów (tych samych dla systemu przed i po integracji)
takich, które nie ulegną procesowi cachowania do bazy i porównaliśmy, jak zmienia się czas generowania odpowiedzi po tej integracji.  
Uśredniony czas dla systemu przed integracją wyniósł ok. 14.66 s, natomiast po ok. 16.15 s. Jak widać, czas generowania 
odpowiedzi przy użyciu naszych modeli SVM zwiększył się. Zakładamy jednak, że jest to przede wszystkim spowodowane faktem, 
że modele SVM są ciężkie, a ich wykorzystanie jest kosztowne czasowo.

### 4. Wnioski

### 5. Linki