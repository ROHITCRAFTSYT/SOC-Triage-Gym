# Base image pinned by digest for reproducible builds; Dependabot (docker
# ecosystem) proposes digest bumps weekly.
FROM python:3.11-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SOC_GYM_LOG_FORMAT=json \
    SOC_GYM_LOG_LEVEL=INFO

WORKDIR /app

# Install dependencies (--prefer-binary avoids Rust/C compilation)
COPY server/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
# gradio (via openenv-core) still pins pillow<13, but every 12.x release carries
# the 2026 Pillow heap-overflow advisories. Lift it past the pin (mirrors the
# uv override in pyproject.toml); pip's warning about the gradio pin is expected.
RUN pip install --no-cache-dir --prefer-binary --no-deps "pillow>=13.0.0"

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
#   SOC_GYM_LOG_FORMAT         json (default in this image) | text
#   SOC_GYM_LOG_LEVEL          DEBUG | INFO (default) | WARNING

# Healthcheck uses the interpreter already in the image instead of installing
# curl: one less package (and CVE feed) in the runtime image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"]

# --no-access-log: the app's RequestLoggingMiddleware emits structured access
# lines with request IDs; uvicorn's plain-text access log would duplicate them.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
