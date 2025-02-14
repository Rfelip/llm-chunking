FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install curl and certificates for downloading UV
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and install UV package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure UV is in the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Copy necessary files before syncing dependencies
COPY pyproject.toml uv.lock .

# Install dependencies using UV
RUN uv sync --frozen

RUN uv pip install llama-cpp-python
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY ./app .  

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300", "--timeout-graceful-shutdown", "300"]
