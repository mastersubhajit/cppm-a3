# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline setup for the Car Price Predictor application.

## Pipeline Overview

The CI/CD pipeline consists of three main workflows:

### 1. CI Pipeline (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `master`, `main`, or `develop` branches
- Pull requests to `master` or `main` branches

**Jobs:**

#### Code Quality & Linting
- **Black** formatting check
- **isort** import sorting check  
- **flake8** code linting
- **bandit** security linting
- Uploads security reports as artifacts

#### Run Tests
- Installs dependencies from `requirements.txt`
- Runs basic import tests
- Executes pytest with coverage reporting
- Uploads coverage reports as artifacts

#### Docker Build & Security Scan
- Builds both main application and MLflow Docker images
- Tests container startup functionality
- **Trivy** vulnerability scanning of Docker images
- Uploads security scan results to GitHub Security tab

#### Dependency Security Check
- **Safety** check for known vulnerabilities in Python dependencies
- Uploads security reports as artifacts

### 2. CodeQL Security Scan (`.github/workflows/codeql.yml`)

**Triggers:**
- Push to `master` or `main` branches
- Pull requests to `master` or `main` branches  
- Scheduled weekly scan (Mondays at 2 AM UTC)

**Features:**
- Advanced static analysis security testing
- Python-specific security and quality checks
- Integrates with GitHub Security tab

### 3. Deploy to Production (`.github/workflows/deploy.yml`)

**Triggers:**
- Push to `master` or `main` branches
- Version tags (`v*`)
- Manual workflow dispatch with environment selection

**Features:**
- Builds and pushes Docker images to GitHub Container Registry
- Supports staging and production deployments
- Uses Docker layer caching for faster builds
- Deployment status notifications

## Security Features

### Code Security
- **Bandit**: Identifies common security issues in Python code
- **CodeQL**: Advanced semantic code analysis  
- **Trivy**: Comprehensive vulnerability scanner for containers
- **Safety**: Checks Python dependencies for known security vulnerabilities

### Container Security
- Multi-stage Docker builds for minimal attack surface
- Regular base image updates
- Vulnerability scanning of final images
- Security scan results integrated with GitHub Security tab

## Code Quality Standards

### Formatting & Style
- **Black**: Automatic code formatting (line length: 127 characters)
- **isort**: Import statement organization
- **flake8**: PEP 8 compliance checking with custom configuration

### Testing
- **pytest**: Test framework with coverage reporting
- Basic structure and import validation tests
- Configurable test discovery and execution

## Configuration Files

- **`pyproject.toml`**: Python project configuration including Black, isort, pytest, and coverage settings
- **`.flake8`**: Flake8 linting configuration with exclusions and ignore rules
- **`.gitignore`**: Updated to exclude CI/CD artifacts and tool caches

## Usage

### Running Tests Locally
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_basic.py -v
```

### Code Quality Checks
```bash
# Install code quality tools
pip install black isort flake8 bandit safety

# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Security scan
bandit -r .

# Dependency check
safety check --file=requirements.txt
```

### Docker Operations
```bash
# Build main application
docker build -t cppm-app .

# Build MLflow service  
docker build -f mlflow.Dockerfile -t cppm-mlflow .

# Run with docker-compose
docker-compose up
```

## Monitoring & Reporting

### Artifacts & Reports
- **Bandit Security Report**: JSON format security analysis
- **Safety Report**: Dependency vulnerability analysis
- **Coverage Reports**: HTML and XML test coverage reports
- **Trivy SARIF**: Container vulnerability scan results

### Integration Points
- GitHub Security tab for vulnerability findings
- Artifact storage for all security and quality reports
- Status checks on pull requests
- Deployment notifications

## Best Practices

### Development Workflow
1. Create feature branch
2. Make changes following code quality standards
3. Run tests locally before committing
4. Create pull request
5. CI pipeline automatically validates changes
6. Merge only after all checks pass

### Security Guidelines
- Regularly update dependencies
- Monitor security scan results
- Address high-severity vulnerabilities promptly  
- Review and approve workflow changes carefully

### Deployment Process
1. Changes merged to main branch trigger automatic staging deployment
2. Version tags (v*) trigger production deployment
3. Manual deployment available via workflow dispatch
4. Container images versioned and stored in registry

## Future Enhancements

- Integration tests with live application endpoints
- Performance benchmarking and monitoring
- Multi-environment deployment automation
- Slack/Teams integration for notifications
- Automated dependency updates with Dependabot