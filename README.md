# 🚀 CPPM-A3: Continuous Pipeline for Predictive Modeling (A3)

![Build](https://github.com/mastersubhajit/cppm-a3/actions/workflows/testing.yml/badge.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A fully automated **Machine Learning Model Deployment and Promotion System** for predictive car price classification — built with **MLflow**, **Docker Compose**, **Pytest**, and **GitHub Actions**.

This project implements an **MLOps pipeline** that continuously tests, deploys, and promotes ML models from *Staging* to *Production* environments through a secure and containerized setup.

---

## 📘 **Project Overview**

`cppm-a3` (Car Price Prediction Model – A3) is part of a continuous model deployment workflow designed for:
- Automated **model training**, **tracking**, and **versioning** in MLflow.
- Containerized application and inference using **Docker Compose**.
- Continuous Integration and Testing using **GitHub Actions**.
- Seamless **Staging → Production** model transition once validation passes.

### 🔍 Key Components
| Component | Description |
|------------|-------------|
| `pages/a3_model.py` | Dash web app for A3 model inference and UI |
| `tests/test_model_staging.py` | Model validation tests (A1, A2, A3) before promotion |
| `tests/test_app_callbacks.py` | UI callback and integration tests |
| `utils.py` | MLflow model loader, scaler utilities, and promotion logic |
| `transition.py` | Entry script to promote model from Staging → Production |
| `.github/workflows/testing.yml` | CI/CD pipeline definition |
| `Dockerfile` / `docker-compose.yml` | Container build and orchestration files |

---

## ⚙️ **Tech Stack**

- **Python 3.12**
- **MLflow Tracking & Registry**
- **Scikit-learn**
- **Dash (Plotly)**
- **Docker & Docker Compose**
- **GitHub Actions (CI/CD)**
- **Pytest** for unit and staging tests
- **Cloudpickle** for custom serialization

---

## 🧠 **Pipeline Workflow**

### 1️⃣ **Build & Test (CI)**
Triggered automatically when a new pre-release version tag (e.g. `v2.2.50`) is pushed.

Steps:
- Build the Docker image using `docker-compose build`.
- Spin up containers with `docker-compose up -d`.
- Run **pytest** inside the container to validate models and app callbacks.
- On success → push the built image to **Docker Hub**.

### 2️⃣ **Deploy (CD)**
- SSH into the remote production server using `appleboy/ssh-action`.
- Pull the latest image.
- Restart the production containers.
- Transition MLflow model version from **Staging → Production**.

---

## 🧩 **Environment Variables**

| Variable | Description | Example |
|-----------|-------------|----------|
| `MLFLOW_TRACKING_URI` | URL to MLflow tracking server | `https://mlflow.ml.brain.cs.ait.ac.th` |
| `APP_MODEL_NAME` | Registered model name in MLflow | `st125998-a3-model` |
| `MLFLOW_TRACKING_USERNAME` | MLflow username | `admin` |
| `MLFLOW_TRACKING_PASSWORD` | MLflow password | `password` |
| `DOCKERHUB_USERNAME` | DockerHub username | *(GitHub Secret)* |
| `DOCKERHUB_TOKEN` | DockerHub access token | *(GitHub Secret)* |

---

## 🧪 **Testing**

Run tests locally with Docker:

```bash
docker compose build
docker compose up -d
docker compose exec -T cppm pytest -v
```
## Structure
```
cppm-a3/
├── .github/workflows/testing.yml      # CI/CD workflow
├── Dockerfile                         # Image build
├── docker-compose.yml                 # Container orchestration
├── models/                            # Serialized scalers & models
├── pages/
│   ├── a3_model.py                    # Dash app
├── tests/
│   ├── test_model_staging.py          # Model validation tests
│   ├── test_app_callbacks.py          # UI callback tests
├── utils.py                           # MLflow + model utils
├── transition.py                      # Promotion script
├── data/
│   └── Cars.csv                       # Dataset for validation
└── README.md
```
