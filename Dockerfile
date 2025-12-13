# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Docling
# OpenGL libraries needed for OpenCV/image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create necessary directories for Chainlit
RUN mkdir -p .chainlit .files && \
    chmod 777 .chainlit .files

# Expose Chainlit default port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
