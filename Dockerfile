# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Assuming requirements.txt is generated, but we can also just install directly
# For production, we'd copy a real requirements.txt, but here we just pip install
RUN pip install --user --no-warn-script-location fastapi uvicorn pydantic networkx torch torch_geometric onnx onnxruntime requests httpx websockets

# Final stage
FROM python:3.11-slim

WORKDIR /app
# Copy installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application code
COPY core /app/core
COPY api /app/api
COPY deploy /app/deploy

# Expose API port
EXPOSE 8000

# Healthcheck using python instead of curl (since curl isn't in the slim image)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Default command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
