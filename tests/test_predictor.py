"""
Tests for the CarPricePredictor class.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.cppm_a3.predictor import CarPricePredictor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "brand": ["Toyota", "Honda", "BMW", "Toyota", "Honda", "BMW"],
            "fuel_type": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel"],
            "mileage": [15.5, 18.2, 12.8, 16.1, 17.3, 13.2],
            "engine_size": [1.6, 1.8, 2.0, 1.5, 1.7, 2.5],
            "price_category": ["Low", "Medium", "High", "Low", "Medium", "High"],
        }
    )


@pytest.fixture
def predictor():
    """Create a CarPricePredictor instance."""
    return CarPricePredictor(random_state=42)


class TestCarPricePredictor:
    """Test cases for CarPricePredictor."""

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.random_state == 42
        assert not predictor.is_fitted
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_fit(self, predictor, sample_data):
        """Test model fitting."""
        predictor.fit(sample_data, "price_category")
        assert predictor.is_fitted

    def test_predict_before_fit_raises_error(self, predictor, sample_data):
        """Test that prediction before fitting raises an error."""
        test_data = sample_data.drop(columns=["price_category"])
        with pytest.raises(
            ValueError, match="Model must be fitted before making predictions"
        ):
            predictor.predict(test_data)

    def test_predict_proba_before_fit_raises_error(self, predictor, sample_data):
        """Test that predict_proba before fitting raises an error."""
        test_data = sample_data.drop(columns=["price_category"])
        with pytest.raises(
            ValueError, match="Model must be fitted before making predictions"
        ):
            predictor.predict_proba(test_data)

    def test_predict(self, predictor, sample_data):
        """Test model prediction."""
        predictor.fit(sample_data, "price_category")
        test_data = sample_data.drop(columns=["price_category"])
        predictions = predictor.predict(test_data)

        assert len(predictions) == len(test_data)
        assert all(
            pred in sample_data["price_category"].unique() for pred in predictions
        )

    def test_predict_proba(self, predictor, sample_data):
        """Test model probability prediction."""
        predictor.fit(sample_data, "price_category")
        test_data = sample_data.drop(columns=["price_category"])
        probabilities = predictor.predict_proba(test_data)

        assert probabilities.shape[0] == len(test_data)
        assert probabilities.shape[1] == len(sample_data["price_category"].unique())
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_save_load_model(self, predictor, sample_data):
        """Test model saving and loading."""
        predictor.fit(sample_data, "price_category")

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save model
            predictor.save_model(filepath)
            assert os.path.exists(filepath)

            # Load model
            loaded_predictor = CarPricePredictor.load_model(filepath)
            assert loaded_predictor.is_fitted
            assert loaded_predictor.random_state == predictor.random_state

            # Test predictions are the same
            test_data = sample_data.drop(columns=["price_category"])
            original_predictions = predictor.predict(test_data)
            loaded_predictions = loaded_predictor.predict(test_data)

            np.testing.assert_array_equal(original_predictions, loaded_predictions)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_before_fit_raises_error(self, predictor):
        """Test that saving before fitting raises an error."""
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp_file:
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                predictor.save_model(tmp_file.name)

    def test_preprocess_data(self, predictor, sample_data):
        """Test data preprocessing."""
        X, y = predictor.preprocess_data(sample_data, "price_category")

        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(sample_data.columns) - 1  # Excluding target column
        assert len(y) == len(sample_data)
        assert (
            not predictor.is_fitted
        )  # Should not change fitted status during preprocessing
