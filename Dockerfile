FROM python:3.9-slim

# Install Tesseract and system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD gunicorn app:app --workers 4 --bind 0.0.0.0:$PORT
