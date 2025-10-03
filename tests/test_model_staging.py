"""
Test all models (A1, A2, A3) to ensure they work correctly.
If tests pass, models are ready for production deployment.
"""
import os
import numpy as np
import pandas as pd
import pytest
import joblib
import mlflow
import cloudpickle
import warnings
import urllib3

# Suppress SSL and sklearn warnings in CI/CD
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Cars.csv")

# Fixed feature ordering to match a3_model.py
NUMERIC_COLS_ORDER = ['year', 'max_power', 'mileage', 'engine']
BRAND_LIST = [
    'Ambassador','Ashok','Audi','BMW','Chevrolet','Daewoo','Datsun','Fiat',
    'Force','Ford','Honda','Hyundai','Isuzu','Jaguar','Jeep','Kia','Land',
    'Lexus','MG','Mahindra','Maruti','Mercedes-Benz','Mitsubishi','Nissan',
    'Opel','Peugeot','Renault','Skoda','Tata','Toyota','Volkswagen','Volvo'
]
ALL_FEATURES = NUMERIC_COLS_ORDER + [f'brand_{b}' for b in BRAND_LIST]

@pytest.fixture(scope="module")
def a1_model():
    return joblib.load(os.path.join(MODELS_DIR, "cppm_a1_model.pkl"))

@pytest.fixture(scope="module")
def a1_scaler():
    return joblib.load(os.path.join(MODELS_DIR, "cppm_a1_scaler.pkl"))

@pytest.fixture(scope="module")
def a2_model():
    return joblib.load(os.path.join(MODELS_DIR, "cppm_a2_model.pkl"))

@pytest.fixture(scope="module")
def a2_scaler():
    return joblib.load(os.path.join(MODELS_DIR, "cppm_a2_scaler.pkl"))

@pytest.fixture(scope="module")
def a3_model():
    mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
    return mlflow.pyfunc.load_model(model_uri="models:/st125998-a3-model/24")

@pytest.fixture(scope="module")
def a3_scaler():
    with open(os.path.join(MODELS_DIR, "cppm_a3_scaler.pkl"), "rb") as f:
        return cloudpickle.load(f)

@pytest.fixture(scope="module")
def sample_data():
    return pd.read_csv(DATA_PATH).iloc[0:1]

# ------------------ A1 Model Tests ------------------
def test_a1_model_load(a1_model, a1_scaler):
    assert a1_model is not None
    assert a1_scaler is not None

def test_a1_model_prediction(a1_model, a1_scaler):
    X_df = pd.DataFrame([[2019, 94.5, 14.6]], columns=['year', 'max_power', 'mileage'])
    X_scaled = a1_scaler.transform(X_df)
    pred = a1_model.predict(X_scaled)
    assert pred is not None and len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number))

# ------------------ A2 Model Tests ------------------
def test_a2_model_load(a2_model, a2_scaler):
    assert a2_model is not None
    assert a2_scaler is not None

def test_a2_model_prediction(a2_model, a2_scaler):
    X_df = pd.DataFrame([[2019, 94.5, 14.6]], columns=['year', 'max_power', 'mileage'])
    X_scaled = a2_scaler.transform(X_df)
    pred = a2_model.predict(X_scaled)
    assert pred is not None and len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number))

# ------------------ A3 Model Tests ------------------
def test_a3_model_load(a3_model, a3_scaler):
    assert a3_model is not None
    assert a3_scaler is not None

def test_a3_model_prediction(a3_model, a3_scaler):
    # Feature definitions must match training
    NUMERIC_COLS_ORDER = ['year', 'max_power', 'mileage', 'engine']
    BRAND_LIST = [
        'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 
        'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 
        'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 
        'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
    ]
    ALL_FEATURES = NUMERIC_COLS_ORDER + [f'brand_{b}' for b in BRAND_LIST]

    # Build sample input (must include all features)
    X_input_dict = {feat: 0 for feat in ALL_FEATURES}
    X_input_dict.update({
        'year': 2019,
        'engine': 1197.0,
        'max_power': 94.5,
        'mileage': 14.6,
        'brand_Maruti': 1
    })
    X_df = pd.DataFrame([X_input_dict], columns=ALL_FEATURES)

    # --- Scale numeric columns (align names with scaler) ---
    numeric_df = X_df[NUMERIC_COLS_ORDER].copy()
    numeric_df.columns = a3_scaler.feature_names_in_  # align with training names
    X_df[NUMERIC_COLS_ORDER] = a3_scaler.transform(numeric_df)

    # --- Convert to numpy before prediction ---
    X_array = X_df.to_numpy().astype(np.float64)
    pred = a3_model.predict(X_array)

    assert pred is not None
    assert len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number, str, np.str_))
