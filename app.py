from flask import Flask, request, render_template_string, jsonify
import cv2
import numpy as np
import os
from ocr_processor import OCRProcessor, MedicalDataExtractor

app = Flask(__name__)
ocr_processor = OCRProcessor()
data_extractor = MedicalDataExtractor()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<title>Medical OCR</title>
<h2>Upload a Medical Prescription Image</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if data %}
<h3>Extracted Data:</h3>
<pre>{{ data | tojson(indent=2) }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def upload_image():
    data = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            extracted_text = ocr_processor.extract_text_from_image(image)
            data = data_extractor.extract_medical_data(extracted_text)

    return render_template_string(HTML_TEMPLATE, data=data)

@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
