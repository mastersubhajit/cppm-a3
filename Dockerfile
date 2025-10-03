FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment settings
# ENV UV_LINK_MODE=copy
ENV MLFLOW_TRACKING_INSECURE_TLS=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# # Create virtual environment and install dependencies
# RUN uv venv .venv --clear && \
#     uv pip install -r requirements.txt

# Copy project files first
COPY . /app

# Install dependencies directly
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'dash[testing]' pytest pytest-depends
RUN pip install scikit-learn==1.7.1
# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8080 80
# Default command: keep container running
CMD ["python", "sleep", "infinity", "app.py"]