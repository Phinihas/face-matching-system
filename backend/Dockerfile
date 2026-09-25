FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# ── Copy ONLY the wheel file — no source code ──
COPY dist/*.whl /tmp/

# Install the wheel + all dependencies
RUN uv pip install --system --no-cache /tmp/*.whl

# Runtime directories
RUN mkdir -p /app/logs /app/temp_uploads /app/pretrained

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8050

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]