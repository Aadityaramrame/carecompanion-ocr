import os
import cv2
from flask import Flask, request, jsonify
from ocr_processor import OCRProcessor, MedicalDataExtractor

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

# Initialize processors
ocr_processor = OCRProcessor()
data_extractor = MedicalDataExtractor()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the OCR API. Use POST /extract_medical_data to extract data."})

@app.route('/extract_medical_data', methods=['POST'])
def api_extract_medical_data():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Save uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({'error': 'Failed to read uploaded image'}), 400

    # Perform OCR
    try:
        raw_text = ocr_processor.extract_text_from_image(image)
    except Exception as e:
        return jsonify({'error': f'OCR error: {e}'}), 500
    if not raw_text.strip():
        return jsonify({'error': 'OCR returned empty text'}), 400

    # Extract structured data
    try:
        structured_data = data_extractor.extract_medical_data(raw_text)
    except Exception as e:
        return jsonify({'error': f'Data extraction error: {e}'}), 500

    return jsonify(structured_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
