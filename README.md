# BHL Solution — NLP Prompt Routing & Fast Caching Engine

An intelligent NLP prompt routing and semantic caching engine extended from the Best Hacking League (BHL) hackathon project. Developed for the Introduction to Artificial Intelligence (WSI) course at Warsaw University of Technology (WUT).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)
![Framework](https://img.shields.io/badge/fastapi-0.110%2B-009688.svg)
![ML Library](https://img.shields.io/badge/scikit--learn-SVM-f7931e.svg)
![Embeddings](https://img.shields.io/badge/SentenceTransformers-All--MiniLM--L6--v2-ff6f00.svg)
![LLM Integration](https://img.shields.io/badge/LLM-Google%20Gemini-4285f4.svg)

---

## 🚀 Overview

**BHL Solution** optimizes Large Language Model (LLM) interaction pipelines by combining **domain topic classification**, **intent detection (Fact vs. Generative)**, and **vector-based semantic caching**. By routing incoming prompts through lightweight, high-precision SVM classifiers before hitting expensive LLM endpoints (Google Gemini), the system dramatically reduces latency, cuts API token costs, and improves response accuracy.

---

## ✨ Key Features

### 🎯 Multi-Model Topic Classification
- **Domain Classifiers**: Separate Linear SVM models trained on large-scale StackExchange corpora (Biology, Mathematics, Programming).
- **One-vs-Rest Architecture**: Evaluates topic confidence; routes prompts with $>90\%$ confidence to specialized domain handlers or defaults to `"General"`.
- **TF-IDF Vectorization**: High-speed n-gram text representation with custom stop-word filtering.

### ❓ Intent Detection (Fact vs. Generative)
- **Classifier**: Distinguishes between simple factual questions (low-cost lookup) and complex generative prompts requiring creative LLM reasoning.
- **Dataset Curation**: Multi-stage NLP preprocessing pipeline regex-matching ~15,000 prompts down to 4,000 balanced instances.

### ⚡ Fast Semantic Caching & FastAPI Server
- **Vector Database**: Embeds user queries using `SentenceTransformerEmbedder` for fast similarity search.
- **REST & WebSocket API**: Real-time response streaming via FastAPI endpoints (`/prompt` and `/ws/chat`).
- **Cache Bypass Option**: Toggle `skip_cached` dynamically per request.

---

## 📊 Performance & Benchmarks

| Model Domain | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Biology Classifier** | **96.48%** | 0.9648 | 0.9648 | 0.9648 |
| **Mathematics Classifier** | **92.99%** | 0.9300 | 0.9299 | 0.9299 |
| **Programming Classifier** | **92.99%** | 0.9300 | 0.9299 | 0.9299 |
| **Fact vs. Generative** | **95.49%** | 0.9550 | 0.9549 | 0.9549 |

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **API Framework**: FastAPI, Uvicorn, WebSockets, Pydantic
- **Machine Learning**: `scikit-learn` (LinearSVC, TfidfVectorizer), `SentenceTransformers`
- **Data & Processing**: Pandas, NumPy, JSONL, Regex, Pickle
- **LLM Integration**: Google Gemini API (`google-generativeai`)

---

## 💻 Getting Started

### Prerequisites

- **Python 3.10** or higher
- `virtualenv` recommended

### Installation & Setup

1. **Clone the repository & create environment**:
   ```bash
   git clone https://github.com/your-username/BHL-solution.git
   cd BHL-solution
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Launch the FastAPI Server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - OpenAPI Documentation will be available at `http://localhost:8000/docs`.

---

## 🏋️ Model Training & Datasets

To retrain the topic or intent classifiers from scratch:

1. Download dataset archives and extract into `train/datasets_preprocessing/`.
2. Run data cleaning pipelines:
   ```bash
   jupyter notebook train/datasets_preprocessing/datasets_preprocessing.ipynb
   ```
3. Run model training notebooks:
   - `train/three_models_training/train_three_models.ipynb`
   - `train/one_model_training/svm_training_one-model.ipynb`
4. Saved `.pkl` model artifacts will be placed under `train/saved_models/`.

---

## 📂 Project Structure

```
BHL-solution/
├── app/
│   ├── main.py                         # FastAPI web server & WebSocket endpoints
│   ├── requirements.txt                # Python dependencies
│   ├── handler/                        # Prompt routing & pipeline orchestrator
│   ├── database/                       # Vector DB & SentenceTransformer embedder
│   ├── topic_classifiers/              # SVM Topic classifier wrappers
│   └── prompts_classification/         # Fact vs. Generative intent classifier
├── train/
│   ├── datasets_preprocessing/         # Dataset cleaning & preprocessing scripts
│   ├── one_model_training/             # Single multi-class SVM training notebooks
│   ├── three_models_training/           # 3x OvR SVM topic model training notebooks
│   └── saved_models/                   # Trained .pkl models & vectorizers
├── NLP_project_raport.md              # Detailed Polish academic research report
└── instruction.md                     # Script execution guidelines
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
