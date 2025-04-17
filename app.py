import os
import tempfile
from flask import Flask, request, jsonify, render_template
from ocr_processor import OCRProcessor

app = Flask(__name__)
ocr_processor = OCRProcessor()

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", error="No selected file")
            
        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type")

        try:
            # Use temporary directory for file processing
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file.filename)
                file.save(file_path)
                
                extracted_text = ocr_processor.extract_text(file_path)
                if not extracted_text:
                    return render_template("index.html", error="OCR failed")
                    
                medical_data = ocr_processor.extract_medical_data(extracted_text)
                return render_template("index.html", data=medical_data)
                
        except Exception as e:
            app.logger.error(f"Processing error: {str(e)}")
            return render_template("index.html", error="Processing error")

    return render_template("index.html")

@app.route("/extract_medical_data", methods=["POST"])
def api_extract_medical_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename)
            file.save(file_path)
            
            extracted_text = ocr_processor.extract_text(file_path)
            if not extracted_text:
                return jsonify({"error": "OCR failed"}), 400
                
            medical_data = ocr_processor.extract_medical_data(extracted_text)
            return jsonify(medical_data)
            
    except Exception as e:
        app.logger.error(f"API Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
