import os
from flask import Flask, request, jsonify
import cv2
from ocr import MedicalOCR
from extractor import MedicalDataExtractor

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr_processor = MedicalOCR()
data_extractor = MedicalDataExtractor()

@app.route("/api/extract", methods=["POST"])
def extract_image_data():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    extracted_text = ocr_processor.extract_text_from_image(image)
    data = data_extractor.extract_medical_data(extracted_text)

    return jsonify({"extracted_data": data})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Medical OCR API is running."})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
