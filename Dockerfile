FROM python:3.12-slim

# Set working directory
WORKDIR /app

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

# Set Matplotlib backend to Agg (headless mode)
RUN pip install --upgrade requests urllib3

# Install uv (fast Python package manager)
RUN pip install uv

# Copy project files
COPY . /app

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    uv pip install --upgrade pip && \
    uv pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080 80
# Default command: keep container running
CMD ["sleep", "infinity", "gunicorn", "--bind", "0.0.0.0:80", "app:server"]