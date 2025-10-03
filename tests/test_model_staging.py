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

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Cars.csv")

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

# A1 Model Tests (Regression: year, max_power, mileage)
def test_a1_model_load(a1_model, a1_scaler):
    assert a1_model is not None
    assert a1_scaler is not None

def test_a1_model_prediction(a1_model, a1_scaler):
    X = np.array([[2019, 94.5, 14.6]])
    X_scaled = a1_scaler.transform(X)
    pred = a1_model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number))

# A2 Model Tests (Regression: year, max_power, mileage)
def test_a2_model_load(a2_model, a2_scaler):
    assert a2_model is not None
    assert a2_scaler is not None

def test_a2_model_prediction(a2_model, a2_scaler):
    X = np.array([[2019, 94.5, 14.6]])
    X_scaled = a2_scaler.transform(X)
    pred = a2_model.predict(X_scaled)
    assert pred is not None
    assert len(pred) == 1
    assert isinstance(pred[0], (int, float, np.number))

# A3 Model Tests (Classification: year, engine, max_power, mileage, brand)
def test_a3_model_load(a3_model, a3_scaler):
    assert a3_model is not None
    assert a3_scaler is not None

def test_a3_model_prediction(a3_model, a3_scaler):
    # Create test data matching A3 model structure - use same order as training
    NUMERIC_COLS_ORDER = ['year', 'max_power', 'mileage', 'engine']
    BRAND_LIST = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'BMW', 'Audi']
    ALL_FEATURES = NUMERIC_COLS_ORDER + [f'brand_{b}' for b in BRAND_LIST]
    
    # Build input
    X_input_dict = {feat: 0 for feat in ALL_FEATURES}
    X_input_dict.update({'year': 2019, 'engine': 1197, 'max_power': 94.5, 'mileage': 14.6, 'brand_Maruti': 1})
    
    X_df = pd.DataFrame([X_input_dict], columns=ALL_FEATURES)
    # Use DataFrame for scaler to preserve feature names - use same order as training
    X_df[NUMERIC_COLS_ORDER] = a3_scaler.transform(X_df[NUMERIC_COLS_ORDER])
    
    pred = a3_model.predict(X_df)
    assert pred is not None
    assert len(pred) == 1