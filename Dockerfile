FROM python:3.12-slim

ENV PYTHONPATH=/opt/engine
WORKDIR /opt/engine

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
# Copy and install requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ ./

# The command to run your application using Uvicorn
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005"]