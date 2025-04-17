import os
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor  # Handles both OCR + data extraction

# Ensure the uploads folder exists
ios.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

# Initialize processor
ocr_processor = OCRProcessor()

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
                # Return JSON error for API clients
                if request.headers.get("Accept") == "application/json":
                    return jsonify({"error": "OCR failed or returned empty text"}), 400
                return render_template("index.html", data=None, error="OCR failed or returned empty text")

            structured_data = ocr_processor.extract_medical_data(extracted_text)

            # Render HTML view
            return render_template("index.html", data=structured_data)

    return render_template("index.html", data=None)

@app.route("/extract_medical_data", methods=["POST"])
def api_extract_medical_data():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Save uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # OCR + Data Extraction
    extracted_text = ocr_processor.extract_text(file_path)
    if not extracted_text:
        return jsonify({"error": "OCR failed or returned empty text"}), 400

    structured_data = ocr_processor.extract_medical_data(extracted_text)
    return jsonify(structured_data)

if __name__ == "__main__":
    # Use PORT env var for platforms like Render; default to 5000
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
