FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies (--prefer-binary avoids Rust/C compilation)
COPY server/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy source code
COPY . .

# Run as a non-root user (production hardening). HF Spaces also runs
# containers unprivileged, so this stays compatible with the Space deploy.
RUN useradd --create-home --uid 1000 socgym \
    && mkdir -p /data/audit \
    && chown -R socgym:socgym /app /data
USER socgym

# HF Spaces requires port 7860
EXPOSE 7860

# Production knobs (all optional, off by default):
#   SOC_GYM_API_KEY            require this key on API requests
#   SOC_GYM_RATE_LIMIT         requests/minute per client (0 = off)
#   SOC_GYM_MAX_SESSIONS       concurrent session cap (default 64)
#   SOC_GYM_SESSION_TTL        idle session eviction, seconds (default 3600)
#   SOC_GYM_AUDIT_DIR          durable JSONL audit export directory
#   SOC_GYM_AUDIT_MAX_EPISODES in-memory audit window (default 200)

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
