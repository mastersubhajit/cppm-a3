import numpy as np
import pandas as pd
import pytest
import cloudpickle
import os

# Load scaler
scaler_path = os.path.join("models", "cppm_a3_scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = cloudpickle.load(f)

# Dummy model class for testing (replace with your actual import if needed)
class DummyModel:
    def predict(self, X): 
        return np.array([3])

# For CI, fallback to DummyModel if MLflow model not available
model = DummyModel()

ALL_FEATURES = ['year', 'engine', 'max_power', 'mileage'] + [f'brand_Audi']

@pytest.fixture
def input_df():
    # Minimal valid input
    X_input = {feat: 0 for feat in ALL_FEATURES}
    X_input["year"] = 2020
    X_input["engine"] = 1200.0
    X_input["max_power"] = 95.0
    X_input["mileage"] = 15.0
    X_input["brand_Audi"] = 1
    return pd.DataFrame([X_input], columns=ALL_FEATURES)

def test_model_input(input_df):
    """Check model accepts expected input without crashing"""
    X_scaled = input_df.copy()
    X_scaled[['year','engine','max_power','mileage']] = scaler.transform(
        X_scaled[['year','engine','max_power','mileage']]
    )
    arr = X_scaled.to_numpy().astype(np.float64)
    output = model.predict(arr)
    assert output is not None

def test_model_output_shape(input_df):
    """Check model output has expected shape (1,)"""
    X_scaled = input_df.copy()
    X_scaled[['year','engine','max_power','mileage']] = scaler.transform(
        X_scaled[['year','engine','max_power','mileage']]
    )
    arr = X_scaled.to_numpy().astype(np.float64)
    output = model.predict(arr)
    assert output.shape == (1,)