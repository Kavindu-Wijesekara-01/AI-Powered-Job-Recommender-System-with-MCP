# AI-Powered Career Intelligence and Real-Time Job Recommendation Platform

An enterprise-grade, microservice-based machine learning application that parses candidate curriculum vitae (PDF), extracts structured profiles via Large Language Models, classifies career domains using trained XGBoost models, and scores real-time LinkedIn job opportunities using multi-criteria ranking algorithms.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Hybrid Decision Pipeline](#hybrid-decision-pipeline)
3. [Multi-Criteria Job Matching Engine](#multi-criteria-job-matching-engine)
4. [MLOps & Reproducibility](#mlops--reproducibility)
5. [User Interface & Presentation](#user-interface--presentation)
6. [Technology Stack](#technology-stack)
7. [Directory Structure](#directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Running the Application](#running-the-application)
10. [API Specification](#api-specification)

---

## System Architecture

The platform operates as a decoupled microservices architecture comprising a high-performance FastAPI inference engine and a Streamlit presentation layer.

```
+-----------------------------------------------------------------------------------+
|                                CLIENT LAYER (Streamlit)                           |
|  - Two-Tone Executive Dashboard (Port 8501)                                       |
|  - Bilingual Engine: English & Sinhala                                            |
|  - Session State Cache & Interactive Score Breakdown                              |
+-----------------------------------------------------------------------------------+
                                         |
                                (REST API / HTTP)
                                         v
+-----------------------------------------------------------------------------------+
|                             BACKEND ENGINE (FastAPI)                              |
|  - Asynchronous Service (Port 8000)                                               |
|  - Document Extraction: PyMuPDF (fitz) & Regex Sanitation                         |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                   HYBRID 3-STAGE INFERENCE PIPELINE                         |  |
|  |  Stage 1: Groq LLM (Structured JSON Extraction)                            |  |
|  |  Stage 2: XGBoost Classifier (Domain Prediction & Calibrated Probability)   |  |
|  |  Stage 3: Business Rules Engine (Tenure Override & 72% Confidence Filter)   |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                    MULTI-CRITERIA JOB RANKING ENGINE                        |  |
|  |  - Semantic Fit: Sentence-Transformers (all-MiniLM-L6-v2)                   |  |
|  |  - Skill Overlap: RapidFuzz Levenshtein Token Matching                     |  |
|  |  - Parity Index: Verified Tenure & Academic Foundation Weighting           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
           |                                                       |
   (Model Artifacts)                                       (External API Calls)
           v                                                       v
+-----------------------------+               +-------------------------------------+
|      MLOPS REPOSITORY       |               |        EXTERNAL CLOUD SERVICES      |
|  - MLflow (sqlite:///db)    |               |  - Groq Cloud LPU API               |
|  - DVC Artifact Tracking    |               |  - Apify LinkedIn Actor Scraper     |
|  - XGBoost Pickled Weights  |               +-------------------------------------+
+-----------------------------+
```

---

## Hybrid Decision Pipeline

The career classification framework applies a multi-stage approach to prevent hallucination and enforce domain integrity.

### Stage 1: LLM-Based Structural Ingestion
Raw text parsed via PyMuPDF is processed by the Groq Cloud inference engine using a low-latency model (`openai/gpt-oss-120b` or `qwen/qwen3.8-27b`). The model converts free-form resume text into a strict, validated JSON schema:
* Verified work experience tenure (years).
* Primary specialization field.
* Academic degree level and education category.
* Technical and professional skills inventory.

### Stage 2: Statistical Classification (XGBoost)
The structured candidate feature vector is evaluated by a trained XGBoost classifier (`job_recommender_model.pkl`). The classifier outputs the target professional domain alongside a calibrated prediction confidence score.

### Stage 3: Deterministic Business Rules Engine
A deterministic rule layer post-processes the model prediction:
* Rule 1 (Tenure Priority Override): If empirical work experience in a verified technical domain exceeds academic stream duration, the professional domain takes precedence.
* Rule 2 (Academic Domain Guard): Enforces foundational discipline boundaries when experience is nascent.
* Rule 3 (Low Confidence Suppression): If the statistical prediction confidence is below 72.0%, job recommendations are safely suppressed to avoid false-positive alignments.

---

## Multi-Criteria Job Matching Engine

Job matching operates over live postings fetched from LinkedIn (via Apify) in real time:

1. Semantic Distance (Dense Retrieval): Uses Sentence-Transformers (`all-MiniLM-L6-v2`) to generate vector embeddings of the candidate profile and job requisition description, calculating cosine similarity.
2. Skill Overlap (Sparse Verification): Utilizes RapidFuzz to compute token set ratio and partial string metrics between candidate skills and required requisition competencies.
3. Tenure and Academic Parity: Normalizes required years of experience and educational minimums against the candidate's verified background.
4. Composite Score: Aggregates the component weights into a normalized 0-100% composite alignment index.

---

## MLOps & Reproducibility

### MLflow Experiment Tracking
Model training is managed within `model_training.ipynb` and tracked via a local SQLite MLflow registry (`mlflow.db`):
* Tracking URI: `sqlite:///mlflow.db`
* Tracked Parameters: `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`.
* Tracked Metrics: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `Log-Loss`.
* Logged Artifacts: Confusion matrices, classification reports, and serialized model files stored under `mlruns/`.

### DVC (Data Version Control)
Large model binaries and local tracking databases are decoupled from Git and managed with DVC:
* Tracked files: `api/models.dvc`, `api/mlflow.db.dvc`, `mlruns.dvc`.
* Configuration: `.dvc/config`.

---

## User Interface & Presentation

The frontend is constructed using Streamlit with an executive dashboard aesthetic:
* Two-Tone Light Palette: Pure white surfaces (`#ffffff`) and soft background slate (`#f8fafc`) paired with an obsidian dark hero header (`#0f172a`).
* Zero-Emoji Policy: Uses clean geometric status indicators, structured cards, and precise corporate typography.
* Bilingual Architecture: Full runtime switching between English and Sinhala with complete translation coverage for all labels, metrics, tabs, and error boundaries.
* Session State Persistence: Switching languages or navigating tabs maintains candidate data without triggering redundant API calls.
* Responsive Grid: Fluid breakpoints optimized for both mobile screens and wide desktop monitors.

---

## Technology Stack

### Backend & Machine Learning
* Python 3.11+
* FastAPI & Uvicorn (Asynchronous REST API)
* XGBoost & Scikit-Learn (Classification Modeling)
* Sentence-Transformers (`all-MiniLM-L6-v2` dense vector embeddings)
* RapidFuzz (String distance & fuzzy token matching)
* Groq Python SDK (Low-latency LPU inference)
* PyMuPDF / fitz (PDF parsing & stream processing)
* Joblib & NumPy (Matrix operations and serialization)

### MLOps & Infrastructure
* MLflow (Experiment tracking, metric logging, model registry)
* DVC (Data Version Control for model artifacts and SQLite checkpoints)
* Docker & Docker Compose (Multi-container orchestration)
* GitHub Actions (Continuous Integration testing)

### Frontend
* Streamlit (Reactive Python web interface)
* Custom CSS & Responsive Media Queries
* Requests (REST client communication)

### External Integrations
* Groq Cloud API
* Apify LinkedIn Jobs Scraper

---

## Directory Structure

```
├── .dvc/                             # DVC internal metadata
├── .github/workflows/ci.yml          # GitHub Actions CI workflow
├── api/
│   ├── models/                       # Versioned model artifacts
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── job_recommender_model.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_metadata.json
│   ├── src/
│   │   ├── helper.py                 # PyMuPDF, Groq LLM, & Rule Engine logic
│   │   ├── job_api.py                # Apify LinkedIn scraping client
│   │   ├── ml_matcher.py             # Sentence-Transformers & RapidFuzz matcher
│   │   └── skills_db.py              # Normalized skills taxonomy
│   ├── Dockerfile                    # FastAPI service container specification
│   ├── main.py                       # FastAPI application entrypoint
│   ├── mlflow.db                     # SQLite database for MLflow experiments
│   └── requirements.txt              # Backend dependencies
├── ui/
│   ├── .streamlit/config.toml        # Streamlit light theme configuration
│   ├── Dockerfile                    # Frontend container specification
│   ├── app.py                        # Streamlit dashboard implementation
│   └── requirements.txt              # UI dependencies
├── docker-compose.yml                # Multi-service container orchestration
├── model_training.ipynb              # Offline training and MLflow tracking pipeline
├── pyproject.toml                    # UV / Pip project metadata
└── README.md                         # Project documentation
```

---

## Installation & Setup

### Prerequisites
* Python 3.11 or higher
* Git
* Docker Desktop (optional for containerized deployment)
* Active API keys for Groq Cloud and Apify

### Environment Configuration
Create a `.env` file in the project root directory:

```env
groq_api_key=your_groq_api_key_here
apify_api_key=your_apify_api_key_here
API_URL=http://localhost:8000
```

---

## Running the Application

### Option 1: Docker Compose (Recommended)

1. Ensure Docker Desktop is active.
2. Build and start the microservice containers:

```bash
docker-compose up --build
```

3. Access the endpoints:
* Streamlit Dashboard: `http://localhost:8501`
* FastAPI Swagger Docs: `http://localhost:8000/docs`

To terminate:
```bash
docker-compose down
```

### Option 2: Local Python Virtual Environment

#### 1. Setup Virtual Environment
```bash
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate

pip install -r api/requirements.txt
pip install -r ui/requirements.txt
```

#### 2. Start Backend API
Open a terminal instance:
```bash
cd api
uvicorn main:app --reload --port 8000
```
Backend will initialize at `http://localhost:8000`.

#### 3. Start Frontend Dashboard
Open a secondary terminal instance:
```bash
cd ui
streamlit run app.py
```
Frontend will automatically open in the browser at `http://localhost:8501`.

---

## API Specification

### 1. Ingest & Analyze Resume
* Endpoint: `POST /api/analyze-resume`
* Payload: Multipart Form Data with key `file` (PDF file).
* Returns: JSON payload containing parsed resume text, executive summary, skill deficits, strategic roadmap, structured candidate profile parameters, and predicted professional domain.

### 2. Search & Rank Job Opportunities
* Endpoint: `POST /api/search-jobs`
* Payload:
```json
{
  "keywords": ["Software Engineer", "Backend Developer"],
  "resume_text": "Extracted text content...",
  "cv_details": { "experience_years": 3.0, "primary_field": "Software Engineering" }
}
```
* Returns: JSON array of ranked LinkedIn requisitions with match percentages, component score breakdowns, matched competencies, and external application URLs.

---

## License
This project is licensed under the MIT License.
