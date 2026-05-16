# Real-Time Financial Fraud Detection System

A production-grade Machine Learning system that detects fraudulent credit card transactions using XGBoost, served via a FastAPI REST API, with a Streamlit dashboard, MLflow experiment tracking, Docker containerization, and CI/CD deployment on AWS.

---

## Live Demo

| Service | URL |
|---|---|
| API Docs | `http://ec2-13-233-142-87.ap-south-1.compute.amazonaws.com:8000/docs` |
| Dashboard | `http://ec2-13-233-142-87.ap-south-1.compute.amazonaws.com:8501` |

---

## The Problem

Banks and fintech companies process millions of transactions daily. Roughly 0.17% are fraudulent — costing the industry billions annually. Rule-based systems miss novel fraud patterns. Human review at scale is impossible.

This system flags suspicious transactions in real time using a trained XGBoost model with sub-100ms inference.

---

## Model Performance

| Metric | Score |
|---|---|
| Model | XGBoost |
| Decision Threshold | 0.3 |
| Test Recall | 85.7% |
| Test Precision | 89.4% |
| Test F1 | 87.5% |

**Why Recall is the north star metric:** Missing actual fraud (false negative) costs real money. A false alarm (false positive) costs a 2-minute customer service call. The threshold of 0.3 was chosen after analyzing the precision-recall tradeoff across multiple thresholds.

---

## Dataset

- **Source:** Kaggle Credit Card Fraud Detection dataset
- **Size:** 284,807 transactions
- **Fraud rate:** 0.17% (highly imbalanced)
- **Features:** Time, V1–V28 (PCA-transformed for privacy), Amount, Class

> The dataset uses PCA-anonymized features for privacy compliance. In production this system would sit downstream of a feature engineering pipeline that produces these vectors from raw transaction data.

---

## System Architecture

```
Raw CSV Data
      ↓
Data Ingestion       → loads, splits (stratified 80/20), saves to artifacts
      ↓
Data Transformation  → RobustScaler on Time & Amount, saves preprocessor.pkl
      ↓
Model Training       → tunes Logistic Regression, Random Forest, XGBoost
      ↓                 selects best model by test recall
MLflow Tracking      → logs all experiments, metrics, params, model artifacts
      ↓
FastAPI REST API     → serves predictions at /predict endpoint
      ↓
Streamlit Dashboard  → demo interface with fraud/legitimate sampling
      ↓
Docker               → containerized (API + Streamlit as separate services)
      ↓
GitHub Actions       → CI/CD pipeline (test → build → push → deploy)
      ↓
AWS ECR + EC2        → Docker image registry + live deployment
```

---

## Tech Stack

| Category | Tools |
|---|---|
| ML & Data | Scikit-learn, XGBoost, Pandas, Numpy, imbalanced-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2, AWS ECR |
| Testing | Pytest, HTTPX |
| Utilities | Dill, python-dotenv |

---

## Project Structure

```
Real-Time-Financial-Fraud-Detection/
├── src/Fraud_Detection/
│   ├── components/
│   │   ├── data_ingestion.py       ← loads and splits raw data
│   │   ├── data_transformation.py  ← scaling and preprocessing
│   │   └── model_training.py       ← trains and selects best model
│   ├── pipeline/
│   │   ├── train_pipeline.py       ← wires all components together
│   │   └── prediction_pipeline.py  ← loads model and runs inference
│   ├── utils/
│   │   └── common.py               ← save/load objects, evaluate models
│   ├── logger/
│   │   └── logger.py               ← timestamped file + console logging
│   └── exception/
│       └── exceptions.py           ← custom exception with file + line info
├── api/
│   └── main.py                     ← FastAPI application
├── tests/
│   └── test_api.py                 ← pytest test suite
├── research/
│   └── experiment.ipynb            ← EDA and model experimentation
├── app.py                          ← Streamlit dashboard
├── main.py                         ← training pipeline entry point
├── Dockerfile                      ← container definition
├── compose.yml                     ← multi-service Docker setup
├── requirements.txt                ← all dependencies
└── .github/workflows/ci.yml        ← CI/CD pipeline
```

---

## Key Technical Decisions and Why

**RobustScaler over StandardScaler**
Fraud amounts are extreme outliers. RobustScaler uses median and IQR instead of mean and std — making it resistant to those extremes.

**Threshold 0.3 instead of default 0.5**
Analysis across thresholds 0.2–0.9 showed 0.3 gives the best balance for fraud detection where catching fraud matters more than avoiding false alarms.

**No SMOTE in final pipeline**
Initial testing with SMOTE gave 91.8% recall but only 5.8% precision — the model was flagging almost everything as fraud. Removing SMOTE and using class_weight='balanced' gave a healthier tradeoff.

**XGBoost over Random Forest**
Random Forest was too slow for real-time inference. XGBoost with regularization (reg_alpha, reg_lambda) achieved better metrics with faster inference.

---

## What I Learned Building This

### Machine Learning Engineering
- Class imbalance handling — why accuracy is useless for fraud detection and how to use precision, recall, and F1 meaningfully
- Threshold tuning — model outputs are probabilities not binary answers. The decision cutoff is a business choice not a model choice
- The SMOTE trap — SMOTE only on training data, never on test data. Applying it before splitting causes data leakage
- RobustScaler vs StandardScaler — why outlier-resistant scaling matters for financial data
- Train/test stratification — with 0.17% fraud, unstratified splits can produce test sets with very few fraud cases

### Software Engineering
- Python package structure — why every folder needs `__init__.py` and what `find_packages()` does
- Editable installs — `pip install -e .` and why it solves module import errors
- Custom logging — `getLogger` vs `basicConfig`, handlers, formatters, duplicate handler guards
- Custom exceptions — extracting file name and line number from traceback objects
- Dataclass configs — clean way to manage file paths across components
- Git large file mistakes — how to remove committed large files using `git rm --cached`

### API Development
- FastAPI + Pydantic — automatic input validation for all 30 transaction features
- Threshold in API — `predict_proba` gives flexibility, `predict` gives hard decisions. Return both
- DataFrame vs numpy — preprocessors fitted on DataFrames need DataFrames at inference
- Model loading at startup — load once, reuse forever. Never load on every request

### MLOps and Production
- MLflow experiment tracking — logging metrics, params, and model artifacts for every run
- Docker layer caching — copy requirements.txt before code so dependency layers cache correctly
- Docker Compose networking — services talk to each other by service name not localhost
- Environment variables — `os.getenv("KEY", "default")` handles both local and Docker environments
- CI/CD with GitHub Actions — test → build → push → deploy pipeline on every push
- Self-hosted runners — GitHub Actions job running on your own EC2 instance
- AWS ECR — Docker image registry. Build in CI, push to ECR, pull on EC2
- Container cleanup — `docker stop name || true` prevents pipeline failures when container doesn't exist

### System Design
- Data pipeline architecture — ingestion → transformation → training as separate testable components
- Batch vs real-time inference — fraud detection requires real-time (sub-100ms) not batch
- Graceful degradation — API returns clear error if model fails to load instead of crashing
- Port management — separate ports for API (8000) and dashboard (8501)

---

## How to Run Locally

**Prerequisites:** Python 3.12, Docker

```bash
# Clone the repository
git clone https://github.com/JaiswarYash/Real-Time-Financial-Fraud-Detection.git
cd Real-Time-Financial-Fraud-Detection

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Download dataset from Kaggle and place at:
# data/raw/creditcard.csv

# Train the model
python main.py

# Start the API
uvicorn api.main:app --reload

# Start the dashboard (new terminal)
streamlit run app.py
```

**Using Docker:**

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

---

## Running Tests

```bash
pytest tests/ -v
```

---

## CI/CD Pipeline

Every push to `main` triggers:

1. **Continuous Integration** — installs dependencies, runs pytest
2. **Continuous Delivery** — builds Docker image, pushes to AWS ECR
3. **Continuous Deployment** — pulls latest image on EC2, restarts containers

---

## Author

**Yash Kumar**
Early-career AI/ML Engineer building production-grade ML systems.

GitHub: [JaiswarYash](https://github.com/JaiswarYash)