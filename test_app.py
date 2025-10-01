"""
Basic tests for the CPPM application.
"""
import pytest
import sys
import os

# Add the app directory to the path
sys.path.insert(0, '/app')


def test_app_import():
    """Test that the app module can be imported."""
    try:
        import app
        assert app.app is not None, "App instance is None"
        assert app.server is not None, "Server instance is None"
    except Exception as e:
        pytest.fail(f"Failed to import app: {e}")


def test_app_has_pages():
    """Test that the app has pages registered."""
    import app
    import dash
    
    # Check that pages are registered
    assert hasattr(dash, 'page_registry'), "Dash has no page_registry attribute"
    assert len(dash.page_registry) > 0, "No pages registered in the app"


def test_app_layout():
    """Test that the app has a valid layout."""
    import app
    
    assert app.app.layout is not None, "App layout is None"


def test_required_files_exist():
    """Test that required data files exist."""
    required_files = [
        '/app/data/Cars.csv',
        '/app/models/cppm_a1_model.pkl',
        '/app/models/cppm_a1_scaler.pkl',
        '/app/models/cppm_a3_scaler.pkl',
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file missing: {file_path}"
