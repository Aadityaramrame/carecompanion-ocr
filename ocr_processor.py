import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import json
import glob
import os
import logging
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Medical Regex Patterns ---
MEDICAL_PATTERNS = {
    'patient': {
        'age_gender_pattern1': r'PATIENT\s*\(\s*(?P<gender1>M|F|Male|Female)\s*\)\s*/\s*(?P<age1>\d{1,3})(?=Y\b)',
        'age_gender_pattern2': r',\s*(?P<age2>\d{1,3})\s*/\s*(?P<gender2>M|F|Male|Female)\b'
    },
    'patient_extra': {
        'weight': r'Weight\s*\(Kg\)\s*:\s*(\d+)',
        'health_card': r'Health\s*Card[:\s]*Exp[:\s]*(\d{4}[\/\-]\d{2}[\/\-]\d{2})'
    },
    'clinical': {
        'diagnosis': r'(?i)Diagnosis[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine Name)',
        'vitals': {
            'bp': r'(?i)(?:BP|Blood\s*Pressure)[\s:]*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?',
            'pulse': r'(?i)(?:Pulse|Heart\s*Rate)[\s:]*(\d{2,3})\s*(?:bpm)?',
            'temp': r'(?i)(?:Temp|Temperature)[\s:]*(\d{2}\.?\d*)\s*°?[CF]?',
            'rr': r'(?i)(?:RR|Respiratory\s*Rate)[\s:]*(\d{2})\s*(?:/min)?',
            'spo2': r'(?i)(?:SpO2|Oxygen\s*Saturation)[\s:]*(\d{2,3})\s*%?'
        },
        'complaints': r'(?i)Chief\s*Complaints[:\s-]+([\s\S]+?)(?=\n)',
        'reactions': r'(?i)(?:Adverse\s*Reactions)[\s:]+([\s\S]+?)(?=\n)',
        'investigations': r'(?i)(?:Investigations|Tests)[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine|Advice|$)'
    },
    'medications': {
        'pattern': r'(?m)^\s*\d+\)\s*((?:(?!^\s*\d+\)).)+)'
    },
    'advice': r'(?i)Advice[:\s-]+([\s\S]+?)(?=\n\s*(?:Follow\s*Up|Next\s*Visit)|$)',
    'follow_up': r'(?i)Follow\s*Up[:\s-]+(\d{2}[\/\-]\d{2}[\/\-]\d{2,4})'
}


class OCRProcessor:
    """Extracts text from an image using Tesseract OCR."""
    def extract_text_from_image(self, image: np.ndarray) -> str:
        if image is None:
            raise ValueError("No image data provided to OCRProcessor.extract_text")

        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        return text


class MedicalDataExtractor:
    """Extracts structured medical data from OCR text."""
    def extract_age_gender(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        for pattern_key in ['age_gender_pattern1', 'age_gender_pattern2']:
            match = re.search(MEDICAL_PATTERNS['patient'][pattern_key], text)
            if match:
                age = match.group('age1') if 'age1' in match.groupdict() else match.group('age2')
                gender = match.group('gender1') if 'gender1' in match.groupdict() else match.group('gender2')
                return age.strip(), gender.strip()
        return None, None

    def extract_vitals(self, text: str) -> Dict[str, str]:
        vitals = {}
        for vital, pattern in MEDICAL_PATTERNS['clinical']['vitals'].items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip().replace(' ', '')
                vitals[vital] = value
        return vitals

    def clean_lines(self, blob: str) -> List[str]:
        return [line.strip() for line in blob.split('\n') if line.strip()]

    def extract_medical_data(self, text: str) -> Dict:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        result = {
            "patient": {},
            "vitals": {},
            "diagnosis": [],
            "medications": [],
            "investigations": [],
            "advice": [],
            "follow_up": {}
        }

        try:
            age, gender = self.extract_age_gender(text)
            if age and gender:
                result["patient"]["age"] = age
                result["patient"]["gender"] = gender

            weight = re.search(MEDICAL_PATTERNS['patient_extra']['weight'], text, re.I)
            if weight:
                result["patient"]["weight"] = f"{weight.group(1).strip()} kg"

            result["vitals"] = self.extract_vitals(text)

            if (diagnosis := re.search(MEDICAL_PATTERNS['clinical']['diagnosis'], text)):
                result["diagnosis"] = self.clean_lines(diagnosis.group(1))

            if (inv := re.search(MEDICAL_PATTERNS['clinical']['investigations'], text)):
                result["investigations"] = self.clean_lines(inv.group(1))

            meds = re.findall(MEDICAL_PATTERNS['medications']['pattern'], text, re.DOTALL | re.MULTILINE)
            if meds:
                result["medications"].extend([re.sub(r'\s+', ' ', m).strip() for m in meds])

            if (advice := re.search(MEDICAL_PATTERNS['advice'], text)):
                result["advice"] = self.clean_lines(advice.group(1))

            if (follow_up := re.search(MEDICAL_PATTERNS['follow_up'], text)):
                result["follow_up"] = {"date": follow_up.group(1).strip()}

        except Exception as e:
            logging.error(f"Extraction error: {e}")
            return {}

        return result


class MedicalOCRApp:
    """Main application class for processing medical images."""
    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self.ocr_processor = OCRProcessor()
        self.data_extractor = MedicalDataExtractor()

    def get_image_paths(self) -> List[str]:
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        paths = []
        for ext in image_extensions:
            paths.extend(glob.glob(os.path.join(self.input_folder, ext)))
        return paths

    def save_output(self, filename: str, raw_text: str, structured_data: Dict):
        base = os.path.splitext(os.path.basename(filename))[0]
        output_folder = os.path.join(self.input_folder, "outputs")
        os.makedirs(output_folder, exist_ok=True)

        with open(os.path.join(output_folder, f"{base}_text.txt"), 'w', encoding='utf-8') as f:
            f.write(raw_text)

        with open(os.path.join(output_folder, f"{base}_structured.json"), 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

    def process_images(self):
        image_paths = self.get_image_paths()

        if not image_paths:
            logging.warning(f"No images found in {self.input_folder}")
            return

        logging.info(f"Found {len(image_paths)} images.")

        for idx, img_path in enumerate(image_paths, 1):
            logging.info(f"Processing Image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
            try:
                img = cv2.imread(img_path)
                extracted_text = self.ocr_processor.extract_text_from_image(img)
                structured_data = self.data_extractor.extract_medical_data(extracted_text)

                self.save_output(img_path, extracted_text, structured_data)

                logging.info(f"Successfully processed: {os.path.basename(img_path)}")
            except Exception as e:
                logging.error(f"Error processing {os.path.basename(img_path)}: {str(e)}")

    def run(self):
        print("🩺 Medical Prescription Structured Data Extractor 🩺")
        print(f"📂 Input Folder: '{self.input_folder}'")
        self.process_images()


if __name__ == "__main__":
    folder = input("Enter the path to the folder containing medical images: ").strip()
    if not os.path.isdir(folder):
        logging.error("Invalid folder path. Please check and try again.")
    else:
        app = MedicalOCRApp(folder)
        app.run()
