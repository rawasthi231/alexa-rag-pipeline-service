# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies needed at runtime (e.g., curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system appuser && \
    adduser --system --no-create-home --ingroup appuser appuser

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install application dependencies
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app \
    PORT=5500

# Set Gunicorn command with environment variable expansion in the CMD
ENV GUNICORN_CMD_ARGS="--workers=4 --worker-class=gevent --worker-connections=1000 --timeout=30 --keep-alive=5"

# Set file permissions
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application using Gunicorn with the correct port binding
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 4 --worker-class gevent --worker-connections 1000 --timeout 30 --keep-alive 5 main:app"]
