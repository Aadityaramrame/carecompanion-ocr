FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (Render uses $PORT)
EXPOSE 10000

# Start the Flask app
CMD ["python", "app.py"]
