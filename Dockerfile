# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy only pyproject.toml and poetry.lock first
COPY pyproject.toml poetry.lock ./

# Install Python dependencies using Poetry
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads faiss_indexes

# Make ports available to the outside world
EXPOSE 8000 8501

# Create a script to load environment variables and run both FastAPI and Streamlit
RUN echo '#!/bin/bash\n\
set -a\n\
source .env\n\
set +a\n\
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
poetry run streamlit run app.py --server.port 8501 --server.address 0.0.0.0\n'\
> /app/start.sh

RUN chmod +x /app/start.sh

# Run the script when the container launches
CMD ["/app/start.sh"]