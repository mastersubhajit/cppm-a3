"""
Tests for the package initialization.
"""

import pytest

from src.cppm_a3 import __author__, __description__, __version__


def test_package_metadata():
    """Test package metadata."""
    assert __version__ == "0.1.0"
    assert __author__ == "mastersubhajit"
    assert (
        __description__
        == "Multinomial LogisticRegression with Car Price Prediction in Categorical Data"
    )


def test_import_predictor():
    """Test that the main predictor class can be imported."""
    from src.cppm_a3.predictor import CarPricePredictor

    assert CarPricePredictor is not None
