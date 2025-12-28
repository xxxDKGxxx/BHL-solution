# BHL-solution — Dokumentacja


## Ogólny opis
BHL-solution to prosta usługa API (FastAPI), która przyjmuje zapytanie tekstowe (prompt),
sprawdza czy istnieje już pasująca odpowiedź w lokalnej bazie wektorowej (FAISS),
weryfikuje trafność znalezionej odpowiedzi za pomocą modelu cross-encoder,
a jeśli odpowiedź jest nieadekwatna lub brak trafienia — generalizuje zapytanie i generuje krótką, rzeczową odpowiedź przy użyciu modelu LLM (domyślnie: Google Gemini),
po czym zapisuje parę (zogólniony prompt, odpowiedź) w bazie, aby móc zwracać ją w przyszłości z cache.

W skrócie:
- POST /prompt → zwraca odpowiedź oraz informację, czy pochodziła z cache.
- Cache oparty jest o FAISS + Sentence-Transformers (all-mpnet-base-v2).
- Trafność (relevance) odpowiedzi sprawdzana jest modelem cross-encoder/ms-marco-MiniLM-L6-v2.
- Gdy trzeba, zapytanie jest „zogólniane”, a odpowiedź generowana przez LLM (Gemini).


## Architektura i przepływ
1. Klient wywołuje endpoint POST /prompt z JSON-em: { "prompt": str, "skip_cached": bool }.
2. Handler próbuje: jeśli skip_cached = False — pobrać z bazy najbliższą odpowiedź (wektorowo, FAISS).
3. Jeśli brak wyniku lub trafność (cross-encoder) ≤ próg (domyślnie 0.5), aplikacja:
   - prosi LLM o stworzenie „zogólnionego” promptu,
   - prosi LLM o krótką, rzeczową odpowiedź na ten zogólniony prompt,
   - zapisuje (zogólniony prompt, odpowiedź) do bazy FAISS,
   - zwraca świeżo wygenerowaną odpowiedź z cached=false.
4. W przeciwnym razie zwraca znalezioną odpowiedź z cached=true.

Główne komponenty:
- FastAPI (main.py) — serwer i endpoint.
- Handler (handler/defaulthandler.py) — logika wyboru: cache vs generowanie.
- Generalizacja (handler/generalize_answer.py) — dwa wywołania LLM: tworzenie zogólnionego promptu i krótkiej odpowiedzi.
- Baza (database/database.py) — FAISS IndexFlatL2, wektory z Sentence-Transformers.
- Embedder (database/sentence_transformer_embedder.py) — model all-mpnet-base-v2.
- Sprawdzanie trafności (relevance_checkers/cross_encoder_relevance_checker.py) — CrossEncoder z aktywacją Sigmoid.
- LLM (llms/gemini_llm.py) — Google Gemini (model: gemini-flash-latest).


## Wymagania
- Python 3.10+ (zalecane)
- System Windows/Linux/macOS
- Zależności z requirements.txt (m.in. fastapi, uvicorn, sentence-transformers, faiss, transformers, google-generativeai, torch itp.)
- Dostęp do internetu (pobranie modeli; wywołania LLM)

Uwaga: llms/gemini_llm.py zawiera klucz API „na sztywno”. W środowisku produkcyjnym użyj zmiennych środowiskowych i bezpiecznego przechowywania sekretów. Przykład:
- Ustaw zmienną GEMINI_API_KEY i odczytuj ją w kodzie zamiast wpisywać klucz w plik.


## Instalacja
1. Utwórz i aktywuj wirtualne środowisko (przykład PowerShell/Windows):
   - python -m venv venv
   - .\venv\Scripts\Activate.ps1
2. Zainstaluj zależności:
   - pip install -r requirements.txt

Modele sentence-transformers i cross-encoder zostaną pobrane przy pierwszym użyciu.


## Uruchomienie serwera
- Używając uvicorn:
  - uvicorn main:app --reload --host 0.0.0.0 --port 8000

Po uruchomieniu:
- Dokumentacja interaktywna: http://127.0.0.1:8000/docs
- Schemat OpenAPI: http://127.0.0.1:8000/openapi.json


## API
POST /prompt
- Body (JSON):
  - prompt: string — treść zapytania użytkownika
  - skip_cached: bool — jeśli true, pomija cache i wymusza ścieżkę generowania
- Przykład żądania (curl):
  - curl -X POST "http://127.0.0.1:8000/prompt" -H "Content-Type: application/json" -d "{\"prompt\": \"Co to jest FAISS?\", \"skip_cached\": false}"
- Przykład odpowiedzi:
  - { "result": "FAISS to biblioteka Facebook AI do wyszukiwania podobieństwa...", "cached": true }

Uwaga: cached=true oznacza trafienie w bazie i pozytywną ocenę trafności; cached=false oznacza, że odpowiedź została świeżo wygenerowana.


## Konfiguracja i progi
- Próg trafności w PromptHandler.generate_answer: domyślnie 0.5 (CrossEncoder zwraca ~[0..1]).
- Model LLM: llms/gemini_llm.py — gemini-flash-latest.
- Embedder: database/sentence_transformer_embedder.py — all-mpnet-base-v2.
- Indeks FAISS: IndexFlatL2, inicjalizowany przy pierwszej insercji i trzymany w pamięci procesu.

Możliwe kierunki konfiguracji (do rozbudowy):
- Zmienne środowiskowe na klucze API (np. GEMINI_API_KEY), wybór modeli, próg trafności.
- Trwałe przechowywanie indeksu i dokumentów (serializacja FAISS, pliki/DB) zamiast pamięci RAM.


## Struktura projektu (skrót)
- main.py — FastAPI app i endpoint /prompt
- handler/
  - defaulthandler.py — główny PromptHandler
  - generalize_answer.py — funkcje do generalizacji i generacji krótkiej odpowiedzi
- database/
  - database.py — prosty wrapper na FAISS (insert/get)
  - sentence_transformer_embedder.py — embedder Sentence-Transformers
- relevance_checkers/
  - cross_encoder_relevance_checker.py — weryfikacja trafności (CrossEncoder)
- llms/
  - gemini_llm.py — integracja z Google Gemini
  - bio_llm.py, math_llm.py, __init__.py — dodatkowe/alternatywne LLM-y (jeśli używane)
- interface/ — interfejsy abstrakcyjne (model, handler, embedder, DB, relevance checker)
- tests/, test_main.http — przykłady i testy


## Przegląd działania (krok po kroku)
1. Klient wysyła prompt.
2. System pobiera z FAISS najbliższy wektor oraz odpowiedź.
3. Cross-encoder ocenia dopasowanie pary (prompt, odpowiedź).
4. Jeśli ocena ≤ próg lub skip_cached = true albo brak odpowiedzi:
   - LLM tworzy zogólniony prompt na bazie oryginalnego.
   - LLM generuje krótką, rzeczową odpowiedź.
   - Para (zogólniony prompt, odpowiedź) trafia do FAISS.
5. Odpowiedź trafia do klienta wraz z flagą cached.


## Uruchamianie testów
- Jeżeli dostępne, uruchom pytest w repozytorium:
  - pytest -q
- Możesz także użyć pliku test_main.http (np. w IDE) do szybkich wywołań endpointu.


## Rozwiązywanie problemów
- ImportError przy FAISS: upewnij się, że zainstalowano właściwe koło dla twojej platformy (CPU/GPU).
- Długi czas pierwszego uruchomienia: modele pobierają się przy pierwszym użyciu.
- Klucze API: nie trzymaj ich w repozytorium; używaj zmiennych środowiskowych.


## Licencja
Brak jawnej informacji o licencji w repozytorium. Dodaj plik LICENSE, jeśli chcesz określić zasady użycia.


## Roadmapa (propozycje)
- Trwałość indeksu FAISS i dokumentów.
- Konfiguracja przez .env (modele, progi, klucze API).
- Obsługa wielu przestrzeni nazw (np. dziedziny tematyczne).
- Metryki i logowanie (prometheus + grafana).
- Alternatywne modele LLM i embeddery wybierane w runtime.
