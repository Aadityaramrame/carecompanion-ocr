
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config poppler-utils && apt-get clean
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
