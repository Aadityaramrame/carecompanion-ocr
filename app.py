import os
import cv2
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor, MedicalDataExtractor

# Ensure the uploads folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

# Initialize processors
ocr_processor = OCRProcessor()
data_extractor = MedicalDataExtractor()

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    data = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            error_message = 'No file provided'
        else:
            # Save uploaded file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Read image
            image = cv2.imread(file_path)
            if image is None:
                error_message = 'Failed to read uploaded image'
            else:
                # Perform OCR
                try:
                    extracted_text = ocr_processor.extract_text_from_image(image)
                except Exception as e:
                    error_message = f'OCR error: {e}'
                else:
                    if not extracted_text.strip():
                        error_message = 'OCR returned empty text'
                    else:
                        # Extract structured data
                        try:
                            data = data_extractor.extract_medical_data(extracted_text)
                        except Exception as e:
                            error_message = f'Data extraction error: {e}'

    return render_template('index.html', data=data, error=error_message)

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
