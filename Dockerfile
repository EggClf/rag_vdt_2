FROM python:3.10.4-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the LLM library
RUN git clone https://github.com/hung20gg/llm.git && \
    cd llm && \
    pip install --no-cache-dir -r requirements.txt && \
    cd /app

# Copy application code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV DEVICE=cpu

# Default port
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py", "--device", "cpu", "--host", "0.0.0.0", "--port", "8000"]