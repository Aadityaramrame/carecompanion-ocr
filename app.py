import os
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor  # Handles both OCR + data extraction

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)


# Initialize processor
ocr_processor = OCRProcessor()

@app.route("/")
def home():
    return "🚀 OCR App is running!"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # OCR + Data Extraction
            extracted_text = ocr_processor.extract_text(file_path)
            if not extracted_text:
                return jsonify({"error": "OCR failed or returned empty text"}), 400

            structured_data = ocr_processor.extract_medical_data(extracted_text)

            return render_template("index.html", data=structured_data)

    return render_template("index.html", data=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
