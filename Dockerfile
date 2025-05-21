FROM nvidia/cuda:12.1-base
WORKDIR /app

# Author/Maintainer information
LABEL maintainer="Nik Jois <nikjois@llamasearch.ai>"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .

EXPOSE 8000
CMD ["python", "-m", "openworld_backend.cli"] 