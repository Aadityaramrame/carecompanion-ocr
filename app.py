import os
from flask import Flask, request, jsonify

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

# Initialize OCR and extractor (ensure these are defined somewhere)
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

            return jsonify(structured_data)  # Return data as JSON response

    return '''
        <h1>Welcome to the Medical OCR App</h1>
        <p>Upload a medical document to extract structured data.</p>
        <form method="POST" enctype="multipart/form-data">
            <label for="file">Upload a medical document:</label>
            <input type="file" name="file" accept="application/pdf,image/*" required>
            <button type="submit">Upload</button>
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
