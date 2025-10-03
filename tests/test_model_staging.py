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
    # Use exact same features as training
    # IMPORTANT: NUMERIC_COLS_ORDER must match the order the scaler was trained on
    NUMERIC_COLS_ORDER = ['year', 'max_power', 'mileage', 'engine'] # Match a3_model.py
    BRAND_LIST = [
        'Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 
        'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 
        'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 
        'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
    ]
    
    # ALL_FEATURES must match the columns expected by the final MLflow model
    ALL_FEATURES = NUMERIC_COLS_ORDER + [f'brand_{b}' for b in BRAND_LIST]
    
    # Build input
    X_input_dict = {feat: 0 for feat in ALL_FEATURES}
    # Note the engine value has been added here to be consistent with a3_model.py
    X_input_dict.update({'year': 2019, 'engine': 1197.0, 'max_power': 94.5, 'mileage': 14.6, 'brand_Maruti': 1})
    
    X_df = pd.DataFrame([X_input_dict], columns=ALL_FEATURES)
    
    # --- CRITICAL CHANGE: Scale using DataFrame columns ---
    # The scaler expects a DataFrame/columns if it was fitted that way.
    # a3_scaler is loading a cloudpickle object, which might be a custom scaler 
    # or a pipeline that handles feature names. The safest way is to mirror 
    # the correct implementation from a3_model.py, which scales only the numeric columns 
    # and then updates the DataFrame.
    
    # Extract only the columns the scaler expects (NUMERIC_COLS_ORDER)
    numeric_data_to_scale = X_df[NUMERIC_COLS_ORDER] 
    
    # Transform the numeric data
    X_scaled = a3_scaler.transform(numeric_data_to_scale)
    
    # Update the numeric columns in the full DataFrame with the scaled values
    X_df[NUMERIC_COLS_ORDER] = X_scaled
    
    # The MLflow model expects a NumPy array of all features (scaled numerics + OHE brands)
    pred = a3_model.predict(X_df.values) 
    
    assert pred is not None
    assert len(pred) == 1
    # Add an assertion to check the type of the classification prediction
    assert isinstance(pred[0], (int, float, np.number, str, np.str_))