"""
Basic tests for the Car Price Predictor application.
"""

import pytest
import os
import sys

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_project_structure():
    """Test that essential project files exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    essential_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'mlflow.Dockerfile',
        'docker-compose.yml'
    ]
    
    for file in essential_files:
        file_path = os.path.join(project_root, file)
        assert os.path.exists(file_path), f"Essential file {file} is missing"


def test_requirements_file():
    """Test that requirements.txt is not empty and contains essential packages."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_path = os.path.join(project_root, 'requirements.txt')
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    # Check for essential packages
    essential_packages = ['dash', 'pandas', 'numpy', 'scikit-learn', 'mlflow']
    
    for package in essential_packages:
        assert package.lower() in content.lower(), f"Essential package {package} not found in requirements.txt"


def test_docker_files():
    """Test that Dockerfile and mlflow.Dockerfile exist and are not empty."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dockerfile_path = os.path.join(project_root, 'Dockerfile')
    mlflow_dockerfile_path = os.path.join(project_root, 'mlflow.Dockerfile')
    
    # Check main Dockerfile
    assert os.path.exists(dockerfile_path), "Dockerfile is missing"
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    assert 'FROM python:' in content, "Dockerfile does not specify Python base image"
    assert 'app.py' in content, "Dockerfile does not reference app.py"
    
    # Check MLflow Dockerfile
    assert os.path.exists(mlflow_dockerfile_path), "mlflow.Dockerfile is missing"
    with open(mlflow_dockerfile_path, 'r') as f:
        content = f.read()
    assert 'FROM python:' in content, "mlflow.Dockerfile does not specify Python base image"
    assert 'mlflow' in content, "mlflow.Dockerfile does not reference MLflow"


def test_pages_directory():
    """Test that pages directory exists with required page files."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pages_dir = os.path.join(project_root, 'pages')
    
    assert os.path.exists(pages_dir), "Pages directory is missing"
    
    required_pages = ['landing.py', 'new_model.py', 'old_model.py', 'visualization.py']
    
    for page in required_pages:
        page_path = os.path.join(pages_dir, page)
        assert os.path.exists(page_path), f"Required page {page} is missing"


def test_models_directory():
    """Test that models directory exists with model files."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    assert os.path.exists(models_dir), "Models directory is missing"
    
    # Check that there are .pkl files (model files)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    assert len(model_files) > 0, "No model files (.pkl) found in models directory"


def test_data_directory():
    """Test that data directory exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    assert os.path.exists(data_dir), "Data directory is missing"