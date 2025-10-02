FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment settings
ENV UV_LINK_MODE=copy
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

# Install uv (fast Python package manager)
RUN pip install --upgrade pip && pip install uv
RUN pip install 'dash[testing]' -v
RUN pip install pytest
RUN pip install pytest-depends

# Copy project files
COPY . /app

# Create virtual environment and install dependencies
RUN uv venv .venv --clear && \
    uv pip install -r requirements.txt

# Expose ports
EXPOSE 8080 80
ENTRYPOINT ["pytest"]
# Default command: keep container running
CMD ["python", "sleep", "infinity", "app.py", "-v"]