FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build dependencies and clean up in one step
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    tree \
    && apt-get install -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file and install dependencies first to leverage caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app to the container
COPY . /app

# Expose the port
EXPOSE 5000

