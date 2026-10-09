# ⚡ ChurnRadar AI: Real-Time Customer Retention & Predictive MLOps Engine

> *Transforming customer behavioral signals into actionable retention intelligence with explainable AI and production-grade MLOps.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-5A0FC8?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=flat)
![Evidently](https://img.shields.io/badge/Evidently-Drift%20Detection-6B4FBB?style=flat)
![Neon](https://img.shields.io/badge/Neon-Cloud%20Postgres-00E599?style=flat&logo=neon&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🌐 Live Deployments

| Service | Endpoint | Description |
|---|---|---|
| 🖥️ **Web Dashboard** | [churn-prediction-lxbzq4psrkfqc656owuekf.streamlit.app](https://churn-prediction-lxbzq4psrkfqc656owuekf.streamlit.app/) | Interactive portal for live predictions, SHAP breakdowns & drift audits |
| ⚡ **Inference API** | [churn-prediction-k5ef.onrender.com](https://churn-prediction-k5ef.onrender.com) | Production REST API with token security & sub-second latency |
| 📖 **Interactive Docs** | [churn-prediction-k5ef.onrender.com/docs](https://churn-prediction-k5ef.onrender.com/docs) | Swagger OpenAPI explorer for payload testing |

> 💡 **Notice:** All endpoints (except `/health` and `/`) require header authentication (`X-API-Key`).  
> The free-tier backend sleeps during idle periods and spins up within ~30–60s on the initial ping.

---

## 💡 What is ChurnRadar?

**ChurnRadar** is an end-to-end, enterprise-ready MLOps platform engineered to predict customer churn before it happens. Built on top of a multi-model gradient boosting ensemble (**LightGBM + XGBoost + CatBoost**), the system combines high-precision machine learning with production resilience.

From automated hyperparameter optimization via **Optuna** and lineage tracking in **MLflow**, to zero-downtime Champion/Challenger **A/B routing**, **SHAP** feature attribution, and **Evidently AI** data drift monitoring — ChurnRadar bridges the gap between Kaggle-grade competitive modeling and resilient cloud deployment.

---

## 🏆 Kaggle Benchmark (Playground S6E3)

Tested against the massive [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) benchmark (**594,194 records**):

- **Leaderboard Standing:** Top 28% (Rank **1160 / 4142** global teams)
- **Private ROC-AUC:** **`0.9154`** (within 0.003 of the global #1 spot)
- **Validation ROC-AUC (OOF):** **`0.9162`**

### 🔬 Model Face-Off & Ensemble Synergy

| Model Engine | Out-of-Fold ROC-AUC | Role |
|---|:---:|---|
| **LightGBM** | 0.9147 | Fast leaf-wise tree learner |
| **XGBoost** | 0.9154 | High-regularization depth controller |
| **CatBoost** | 0.9150 | Categorical boundary specialist |
| 🌟 **Tri-Ensemble** | **0.9162** | **Optuna-weighted consensus (500 trials)** |

---

## 🚀 Key Architectural Capabilities

- 🎯 **Tri-Boost Ensemble:** Dynamically combines predictions using Bayesian-optimized weights across 5-fold cross-validation.
- 🔀 **Champion vs. Challenger A/B Engine:** Live routing of inference requests between stable production models and experimental challengers.
- 🔍 **Glass-Box SHAP Explainability:** Translates complex tree decisions into transparent, feature-by-feature impact charts per individual customer.
- 📡 **Automated Data Drift Surveillance:** Continuously audits incoming traffic against the reference population with Evidently AI.
- 🏛️ **Cloud Model Registry:** Neon PostgreSQL-backed tracking of model versions, hyperparameters, and live response latencies.
- 🛡️ **Hardened API Gateway:** Equipped with API-key guards, SlowAPI rate-limiting, non-blocking asynchronous logging, and connection pooling.

## System Architecture

### High-Level Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           Training Layer                            │
│                                                                     │
│          Kaggle Dataset → Preprocessing → Optuna HP Tuning          │
│                                  │                                  │
│                         Stratified K-Fold CV                        │
│                                  │                                  │
│              ┌───────────────────┼───────────────────┐              │
│              ▼                   ▼                   ▼              │
│           LightGBM            XGBoost            CatBoost           │
│              └───────────────────┼───────────────────┘              │
│                                  ▼                                  │
│                    Ensemble Weight Optimization                     │
│                     (500 Optuna Trials on OOF)                      │
│                                  │                                  │
│                                  ▼                                  │
│                      MLflow Experiment Tracking                     │
│                                  │                                  │
│                                  ▼                                  │
│           model_manager.py → .pkl files baked into Docker           │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            Serving Layer                            │
│                                                                     │
│                 FastAPI (Render) — Docker Container                 │
│                                                                     │
│     ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐     │
│     │  /predict  │  │ /explain │  │  /drift   │  │ /benchmark │     │
│     │  A/B route │  │   SHAP   │  │ Evidently │  │  Latency   │     │
│     └────────────┘  └──────────┘  └───────────┘  └────────────┘     │
│     ┌──────────────────────────┐  ┌───────────────────────────┐     │
│     │        /ab-report        │  │      /model-registry      │     │
│     │  Champion vs Challenger  │  │  MLflow Lineage Tracking  │     │                          
│     └──────────────────────────┘  └───────────────────────────┘     │
│                                                                     │
│                                  │                                  │
│                                  ▼                                  │
│                           Neon Postgres                             │
│               (predictions, inputs, metrics, registry)              │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            Frontend Layer                            │
│                                                                      │
│              Streamlit Dashboard (Streamlit Community Cloud)          │
│                                                                      │
│ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────────┐ ┌─────┐ ┌──────────┐ │
│ │ Predict │ │ Explain │ │ Drift │ │ Benchmark │ │ A/B │ │ Registry │ │
│ └─────────┘ └─────────┘ └───────┘ └───────────┘ └─────┘ └──────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Request Flow - `/predict` Endpoint

```text
Client Request
      │
      ▼
API Key Authentication
      │
      ├── ❌ Invalid → 401 Unauthorized
      │
      ▼
Rate Limit Check (slowapi - 10 req/min)
      │
      ├── ❌ Exceeded → 429 Too Many Requests
      │
      ▼
Pydantic Input Validation
      │
      ├── ❌ Invalid → 422 Unprocessable Entity
      │
      ▼
choose_model() - A/B Router
      │
      ├── Champion Model (models/champion/)
      └── Challenger Model (models/challenger/)
            │
            ├── ❌ No models loaded → 503 Service Unavailable
            │
            ▼
   Feature Engineering (preprocess.py)
            │
            ▼
   Weighted Ensemble Inference
   (w_lgb × p_lgb) + (w_xgb × p_xgb) + (w_cat × p_cat)
            │
            ▼
   Generate Prediction ID (uuid4)
            │
            ▼
   JSON Response to Client
            │
            ▼ (background tasks - non-blocking)
   Log Prediction + Raw Input → Neon Postgres
```

## Machine Learning Pipeline

The pipeline trains three gradient boosting models (LightGBM, XGBoost, and CatBoost) on the [Kaggle PS S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) dataset (594,194 rows), and combines them into a weighted ensemble optimized for ROC-AUC.

- **Preprocessing:** Schema validation, categorical encoding, and handling of domain-specific values (e.g., "No internet service")
- **Feature Engineering:** Derived features such as `TENURE × MONTHLY_CHARGES`, spend ratios, service counts, and binary payment indicators
- **Validation:** Stratified K-Fold cross-validation across all models
- **Hyperparameter Tuning:** Optuna used to optimize learning rate, tree depth, and subsampling parameters
- **Ensemble:** Out-of-Fold predictions used to optimize ensemble weights via 500 Optuna trials, combined using weighted averaging
- **Experiment Tracking:** All runs logged in MLflow with parameters, ROC-AUC metrics, and model artifacts

### Inference

Models and ensemble weights are loaded from `.pkl` files baked into the Docker image at build time. The final churn probability is computed as:

```
final_probability = (w_lgb × p_lgb) + (w_xgb × p_xgb) + (w_cat × p_cat)
```

A probability above `0.5` is classified as **churn**.

## API Endpoints

> All endpoints except `/` and `/health` require authentication header.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root |
| `GET` | `/health` | Health check (used by UptimeRobot) |
| `POST` | `/predict` | Real-time churn probability with A/B routing |
| `POST` | `/explain` | SHAP feature contribution scores |
| `GET` | `/drift` | Data drift detection via Evidently AI |
| `GET` | `/drift/report` | Interactive Evidently HTML drift report |
| `GET` | `/benchmark` | Per-model inference timing (min/avg/max/p95) |
| `GET` | `/ab-report` | Champion vs Challenger prediction comparison |
| `GET` | `/model-registry` | Active model versions with lineage info |

> Full interactive API docs available at [`/docs`](https://churn-prediction-mlops.onrender.com/docs)

## Tech Stack

| Category | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| ML | LightGBM, XGBoost, CatBoost, Scikit-learn |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Drift Detection | Evidently AI |
| Database | Neon Postgres (psycopg2, connection pooling) |
| Deployment | Docker, Render, Hugging Face Spaces |
| UI | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Security | API key authentication, slowapi rate limiting, CORS |
| Monitoring | UptimeRobot |

## Project Structure

```

churn-prediction-mlops/
├── app/                                - FastAPI application
│   ├── main.py                         - App entry point, endpoints, lifespan
│   ├── schemas.py                      - Pydantic input validation
│   ├── predict.py                      - Inference logic + A/B routing + SHAP explainability
│   ├── drift.py                        - Evidently drift detection
│   ├── benchmark.py                    - Per-model latency benchmarking
│   ├── registry.py                     - Model registry sync
│   ├── monitoring.py                   - Recent input fetching
│   ├── load_model.py                   - Model loading with caching
│   ├── db.py                           - Postgres connection pool
│   ├── logging.py                      - Prediction + input logging
│   └── config/settings.py              - Environment config
├── models/
│   ├── champion/                       - Active production models (.pkl)
│   └── challenger/                     - Candidate models for A/B testing
├── model_training/
│   ├── model_trainer.py                - Training pipeline
│   ├── hyperparameter_optimizer.py     - Optuna studies
│   ├── model_manager.py                - Export models from MLflow
│   └── mlflow_launcher.py              - MLflow server launcher
├── streamlit_app/                      - Streamlit monitoring dashboard
│   ├── app.py                          - Main entry point
│   ├── pages/                          - Predict, Explain, Drift, Benchmark, A/B, Registry
│   ├── components/                     - Navbar, customer form
│   ├── utils/                          - API calls, plots
│   ├── styles/                         - Global CSS styles
│   ├── config/settings.py              - Environment config
│   ├── .streamlit/config.toml          - Streamlit server config
│   ├── Dockerfile                      - HF Spaces deployment
│   └── requirements.txt                - Streamlit dependencies
├── data/
│   ├── reference.csv                   - Reference dataset for drift detection
├── tests/
│   └── test_api.py                     - pytest test cases
├── utils/
│   ├── preprocess.py                   - Shared feature engineering
│   ├── generate_reference_set.py       - Generates drift reference dataset
│   └── generate_dummy_requests.py      - Drift simulation utility
├── .github/workflows/ci.yml            - GitHub Actions CI/CD
├── .env.example                        - Environment variable template
├── .gitattributes                      - Git LFS and line ending config
├── .gitignore                          - Git exclusions
├── .dockerignore                       - Docker build exclusions
├── Dockerfile                          - FastAPI container
├── docker-compose.yml                  - Local development
├── README.md                           - Project documentation
├── requirements.txt                    - FastAPI dependencies
└── setup.py                            - Package installation for local imports

```

## Local Setup

> Requires a Postgres database (e.g. [Neon](https://neon.tech) free tier). Set `DATABASE_URL` accordingly.

```bash
# Clone repository
git clone https://github.com/gaurav-S8/churn-prediction-mlops.git
cd churn-prediction-mlops

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment variables
cp .env.example .env        # for running FastAPI locally
cp .env.example .env.docker # for running FastAPI via Docker
# Fill in DATABASE_URL and API_KEY in both files

# API_URL tells the Streamlit frontend which FastAPI backend to connect to.
# Local development: `http://localhost:8000`
# Deployed backend (e.g. Render): `https://<your-backend>.onrender.com`

# Start FastAPI via Docker
docker-compose up --build

# OR run FastAPI locally
fastapi dev app/main.py

# Start Streamlit dashboard (separate terminal)
cd streamlit_app
streamlit run app.py
```

> Make sure `.env` and `.env.docker` are filled in before starting. Never commit these files.


## CI/CD Pipeline

This project uses **GitHub Actions** for automated testing and deployment.

On every push to the `deploy` branch:

1. Dependencies are installed in a fresh Ubuntu environment
2. 5 pytest tests are executed covering health, root, payload, response structure, and input validation
3. Render deployment is triggered automatically on successful tests
4. Streamlit UI is synchronized to Hugging Face Spaces

### Pipeline Workflow

```text
GitHub Push (deploy branch)
           │
           ▼
   GitHub Actions
           │
           ▼
  Install Dependencies
           │
           ▼
    Run Pytest Suite
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
Render Deploy   HF Spaces Sync
(FastAPI API)   (Streamlit UI)
     │           │
     └─────┬─────┘
           │
           ▼
   Deployment Complete
```

## Future Improvements

- **Kubernetes deployment:** Horizontal pod autoscaling for high-traffic inference
- **Automated model promotion:** Trigger champion/challenger swap based on live ROC-AUC thresholds
- **Feature store:** Centralized feature registry to eliminate training/serving skew
- **Retraining pipeline:** Automatically retrain when Evidently drift scores exceed threshold
- **Async inference:** Celery + Redis queue for high-throughput non-blocking predictions
- **Prometheus + Grafana:** Add production-grade metrics collection, dashboards, and alerting for API health, inference performance, and system resources.

## Acknowledgements

- Dataset: [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
- Database: [Neon Postgres](https://neon.tech)
- API Hosting: [Render](https://render.com)
- Web UI Hosting: [Streamlit Community Cloud](https://streamlit.io/cloud)

## Author

**Shreya Gaur**

- GitHub: [github.com/shreyagaur-10](https://github.com/shreyagaur-10)

[![GitHub](https://img.shields.io/badge/GitHub-shreyagaur--10-181717?style=flat&logo=github&logoColor=white)](https://github.com/shreyagaur-10)

---

## License

This project is licensed under the [MIT License](LICENSE).
