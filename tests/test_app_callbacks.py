"""
Test app callbacks for all model pages to ensure proper functionality.
"""
import pytest
import numpy as np
import dash
from dash import Dash

# Create a Dash app instance before importing pages
app = Dash(__name__, use_pages=True, pages_folder="")

# Now import the pages
from pages import old_model, new_model, a3_model

def test_old_model_callback():
    result = old_model.predict_price_old(1, 2019, 94.5, 14.6)
    assert "Estimated Selling Price: ₹" in result
    assert result != "Enter details and click Predict."

def test_new_model_callback():
    result = new_model.predict_price_new(1, 2019, 94.5, 14.6)
    assert "Estimated Selling Price: ₹" in result
    assert result != "Enter details and click Predict."

def test_a3_model_callback():
    result = a3_model.predict_price_a3(1, 2019, 1197, 94.5, 14.6, "Maruti")
    assert "Predicted Selling Price Class:" in str(result) or "Error:" in str(result)
    assert result != "Enter details and click Predict."

def test_old_model_no_clicks():
    result = old_model.predict_price_old(0, 2019, 94.5, 14.6)
    assert result == "Enter details and click Predict."

def test_new_model_no_clicks():
    result = new_model.predict_price_new(0, 2019, 94.5, 14.6)
    assert result == "Enter details and click Predict."

def test_a3_model_no_clicks():
    result = a3_model.predict_price_a3(0, 2019, 1197, 94.5, 14.6, "Maruti")
    assert result == "Enter details and click Predict."