FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install curl and certificates required for downloading UV
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and install the UV package manager
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the UV binary is available on PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency management files from the bot folder
COPY ./pyproject.toml ./uv.lock .

# Install dependencies using UV with the frozen lock file.
# UV typically creates a virtual environment (defaulting to /app/.venv)
RUN uv sync --frozen

# Add the virtual environment's bin directory to PATH so that executables (like uvicorn) are found
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code
COPY bot.py .

CMD ["python", "bot.py"]
