# cppm-a3
Multinomial LogisticRegression with Car Price Prediction in Categorical Data

[![CI/CD Pipeline](https://github.com/mastersubhajit/cppm-a3/actions/workflows/ci.yml/badge.svg)](https://github.com/mastersubhajit/cppm-a3/actions/workflows/ci.yml)
[![Security Scan](https://github.com/mastersubhajit/cppm-a3/actions/workflows/security.yml/badge.svg)](https://github.com/mastersubhajit/cppm-a3/actions/workflows/security.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)

## Overview

This project implements a multinomial logistic regression model for car price prediction using categorical data. The model is designed to classify car prices into discrete categories based on various car features.

## Features

- **Multinomial Logistic Regression**: Uses scikit-learn's LogisticRegression with multinomial classification
- **Data Preprocessing**: Automatic handling of categorical variables using Label Encoding
- **Feature Scaling**: StandardScaler for numerical features
- **Model Persistence**: Save and load trained models using joblib
- **Comprehensive Testing**: Full test suite with pytest and coverage reporting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (for testing and code quality)
pip install -r requirements-dev.txt
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mastersubhajit/cppm-a3.git
cd cppm-a3

# Install in development mode
pip install -e .
```

## Usage

### Basic Usage

```python
from src.cppm_a3.predictor import CarPricePredictor
import pandas as pd

# Sample data
data = pd.DataFrame({
    'brand': ['Toyota', 'Honda', 'BMW'],
    'fuel_type': ['Petrol', 'Diesel', 'Petrol'],
    'mileage': [15.5, 18.2, 12.8],
    'engine_size': [1.6, 1.8, 2.0],
    'price_category': ['Low', 'Medium', 'High']
})

# Initialize and train the model
predictor = CarPricePredictor(random_state=42)
predictor.fit(data, 'price_category')

# Make predictions
test_data = data.drop(columns=['price_category'])
predictions = predictor.predict(test_data)
probabilities = predictor.predict_proba(test_data)

# Save the model
predictor.save_model('car_price_model.joblib')

# Load the model later
loaded_predictor = CarPricePredictor.load_model('car_price_model.joblib')
```

## Development

### Code Quality

This project uses several tools to ensure code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html --cov-report=xml

# Run specific test file
pytest tests/test_predictor.py
```

### Code Formatting

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .
```

## CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline using GitHub Actions that provides:

### Continuous Integration (`.github/workflows/ci.yml`)
- **Multi-Python Version Testing**: Tests against Python 3.8, 3.9, 3.10, and 3.11
- **Dependency Caching**: Caches pip dependencies for faster builds
- **Code Quality Checks**:
  - Syntax errors and undefined names detection with flake8
  - Code formatting verification with black
  - Import sorting verification with isort
- **Automated Testing**: Runs the complete test suite with pytest
- **Coverage Reporting**: Generates coverage reports and uploads to Codecov

### Security and Dependency Management (`.github/workflows/security.yml`)
- **Dependency Vulnerability Scanning**: Weekly security scans with Safety
- **Code Security Analysis**: Static security analysis with Bandit
- **Dependency Review**: Automated review of dependency changes in PRs
- **Security Report Artifacts**: Stores security reports for review

### Release and Deployment (`.github/workflows/release.yml`)
- **Automated Release Building**: Triggered on git tags and GitHub releases
- **Package Building**: Creates Python wheel and source distributions
- **Package Validation**: Validates packages before release
- **GitHub Releases Integration**: Uploads packages to GitHub releases
- **Docker Image Creation**: Builds and publishes Docker images to GitHub Container Registry
- **PyPI Publishing**: Ready-to-use PyPI publication (commented out by default)

### Pre-commit Hooks (`.pre-commit-config.yaml`)
For developers who want to run quality checks locally:

```bash
# Install pre-commit
pip install pre-commit

# Set up the git hook scripts
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

### Pipeline Configuration Features
- **Job Dependencies**: Ensures tests pass before building/deploying
- **Conditional Execution**: Different workflows for different events
- **Comprehensive Error Handling**: Detailed reporting and artifact collection
- **Security Best Practices**: Uses latest actions and security scanning
- **Caching Strategy**: Optimized for build speed and resource usage

## Project Structure

```
cppm-a3/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── src/
│   └── cppm_a3/
│       ├── __init__.py           # Package initialization
│       └── predictor.py          # Main prediction logic
├── tests/
│   ├── __init__.py
│   ├── test_init.py              # Package metadata tests
│   └── test_predictor.py         # Predictor class tests
├── .gitignore                    # Git ignore patterns
├── pyproject.toml               # Project configuration
├── setup.cfg                    # Tool configuration
├── setup.py                     # Package setup
├── requirements.txt             # Main dependencies
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure all tests pass and code quality checks succeed
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

The CI/CD pipeline will automatically run tests and code quality checks on your pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
