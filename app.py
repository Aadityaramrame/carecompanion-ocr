import os
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor
from extractor import MedicalDataExtractor

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

# Initialize OCR and extractor
ocr_processor = OCRProcessor()
data_extractor = MedicalDataExtractor()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # OCR Processing
            extracted_text = ocr_processor.extract_text(file_path)
            
            if not extracted_text:
                return jsonify({"error": "OCR failed or returned empty text"}), 400

            # Structured data extraction
            structured_data = data_extractor.extract_medical_data(extracted_text)

            return render_template("index.html", data=structured_data)

    return render_template("index.html", data=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
