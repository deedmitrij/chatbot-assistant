# Use an official lightweight Python image
FROM python:3.11-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set working directory inside container
WORKDIR /app

# Install essential build tools (some python modules require compilation during pip install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Flask API port
EXPOSE 5000

# Command to bootstrap the manager, watch and start Flask
CMD ["python", "main.py"]
