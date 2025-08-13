# Multi-stage build for optimized clinical deployment
FROM python:3.12-slim as base

# Install system dependencies for medical imaging and ML
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 clinicalai && \
    chown -R clinicalai:clinicalai /app
USER clinicalai

# Copy application code with proper structure
COPY --chown=clinicalai:clinicalai src/ ./src/
COPY --chown=clinicalai:clinicalai app/ ./app/
COPY --chown=clinicalai:clinicalai config/ ./config/

# TODO: Copy pre-trained models when available
# COPY --chown=clinicalai:clinicalai models/ ./models/

# Health check for clinical deployment monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/health || exit 1

# Expose port
EXPOSE 8501

# Environment variables for production deployment
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# TODO: Add ML framework optimization flags when models are implemented
# ENV OMP_NUM_THREADS=1
# ENV CUDA_VISIBLE_DEVICES=0

# Command to run the clinical application
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
