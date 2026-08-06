# Minimal image for the MCP Tool-Retrieval Gateway.
# Uses the slim base to keep the image small: it installs only the lean runtime
# (fastapi/uvicorn/numpy/pyyaml/mcp) and defaults to the offline hashing
# embedder, so the container starts instantly with no model download. For
# semantic retrieval in a container, also install sentence-transformers and set
# MCP_ROUTER_EMBEDDER=sentence-transformers (expect a much larger image).

FROM python:3.11-slim

# Don't buffer stdout/stderr (so logs show up promptly) and don't write .pyc.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_ROUTER_EMBEDDER=hashing \
    MCP_ROUTER_CONFIG=/app/config.example.yaml

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY mcp_router ./mcp_router
COPY config.example.yaml ./
COPY examples ./examples

EXPOSE 8000

# --factory tells uvicorn to call app_from_env() to build the app, which reads
# MCP_ROUTER_CONFIG. Override that env var (or bind-mount your own config) to
# point at your servers.
CMD ["uvicorn", "mcp_router.server:app_from_env", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
