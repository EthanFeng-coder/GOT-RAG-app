# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    git \
    lsof \
    coreutils \
    vim \
    && rm -rf /var/lib/apt/lists/*
# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 9000

# Run the FastAPI app with Uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]

