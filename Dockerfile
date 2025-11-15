FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/astrogea

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install distributed processing dependencies
RUN pip install --no-cache-dir \
    dask[complete] dask-kubernetes kubernetes \
    boto3 google-cloud-storage azure-storage-blob pyyaml

# Copy astrogea source code
COPY astrogea/ ./astrogea/
COPY pyproject.toml .

# Install astrogea
RUN pip install -e .

# Create directories
RUN mkdir -p /data /cache /var/log/astrogea

# Set environment variables
ENV ASTROGEA_MODE=kubernetes
ENV PYTHONPATH=/opt/astrogea

# Expose ports
EXPOSE 8786 8787

# Default command
CMD ["python", "-m", "astrogea.worker"]






















