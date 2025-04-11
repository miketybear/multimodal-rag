FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for unstructured and its dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    libmagic1 \
    poppler-utils \
    libxml2-dev \
    libxslt1-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a directory for the database
RUN mkdir -p database

# Expose the Streamlit port
EXPOSE 8555

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8555"]
