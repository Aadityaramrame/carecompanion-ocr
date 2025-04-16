import os
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor
from extractor import MedicalDataExtractor

# Ensure the input folder exists
os.makedirs('uploads', exist_ok=True)

app = Flask(__name__)

class MedicalOCRApp:
    """Main application class for processing medical images."""
    def __init__(self, input_folder='uploads'):
        self.input_folder = input_folder
        self.ocr_processor = OCRProcessor()
        self.data_extractor = MedicalDataExtractor()

    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            extracted_text = self.ocr_processor.extract_text_from_image(img)
            structured_data = self.data_extractor.extract_medical_data(extracted_text)
            return structured_data
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save uploaded file to 'uploads' folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Process the image and extract data
            ocr_app = MedicalOCRApp(input_folder='uploads')
            extracted_data = ocr_app.process_image(file_path)

            # Return the extracted data as JSON
            if extracted_data:
                return render_template("index.html", data=extracted_data)
            else:
                return jsonify({"error": "Failed to extract data from image"}), 400

    # If GET request, just render the upload form
    return render_template("index.html", data=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
