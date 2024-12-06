from flask import Flask, render_template_string, request, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import torch
from paddleocr import PaddleOCR
import cv2
from werkzeug.utils import secure_filename
import re

# Initialize Flask app
app = Flask(__name__)

# Paths and configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
tensorflow_model = load_model('F:/FlaskApp/dataset/m.h5')  # Update to your actual model path
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
ocr = PaddleOCR(use_angle_cls=True, lang='en', cpu_threads=1)

# Class names for TensorFlow model
class_names = {
    0: 'fresh_apple',
    1: 'fresh_banana',
    2: 'fresh_bitter_gourd',
    3: 'fresh_capsicum',
    4: 'fresh_orange',
    5: 'fresh_tomato',
    6: 'stale_apple',
    7: 'stale_banana',
    8: 'stale_bitter_gourd',
    9: 'stale_capsicum',
    10: 'stale_orange',
    11: 'stale_tomato'
}

# Helper functions
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_expiry_date(text):
    """
    Extracts expiry date from the given text using common date formats.
    """
    # Regular expression for common date formats
    date_patterns = [
        r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Expiry Date: 20/07/2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 20/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Expiry Date: 20 MAY 2024
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Expiry Date: 20 MAY 2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Expiry Date: 2024/07/2O24
    r'(?:exp(?:iry)?\.?\s*date\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Expiry Date: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}))',  # Best Before: 2025
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Best Before: 20/07/2024
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 20/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Best Before: 20 MAY 2024
    r'(?:best\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Best Before: 20 MAY 2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Best Before: 2024/07/2O24
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Best Before: 2024/07/20
    r'(?:best\s*before\s*[:\-]?\s*.*?(\d{1,2}\d{2}\d{2}))', 
    r'(?:best\s*before\s*[:\-]?\s*(\d{6}))',
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}[0O]\d{2}))',  # Consume Before: 3ODEC2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{1,2}[A-Za-z]{3,}\d{2}))',  # Consume Before: 30DEC23
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Consume Before: 20/07/2024
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 20/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Consume Before: 20 MAY 2024
    r'(?:consume\s*before\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Consume Before: 20 MAY 2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Consume Before: 2024/07/2O24
    r'(?:consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Consume Before: 2024/07/20
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4}))',  # Exp: 20/07/2024
    r'(?:exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 20/07/2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*\d{4}))',  # Exp: 20 MAY 2024
    r'(?:exp\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp: 20 MAY 2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp: 2024/07/2O24
    r'(?:exp\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2}))',  # Exp: 2024/07/20
    r"Exp\.Date\s+(\d{2}[A-Z]{3}\d{4})",
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 16 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-][0O]\d{2}))',  # Exp. Date: 15/12/2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date: 15 MAR 2O30 (with typo)
    r'(?:exp\s*\.?\s*date\s*[:\-]?\s*.?(\d{2}\s[A-Za-z]{3,}\s*[0O]\d{2}))',  # Exp. Date cdsyubfuyef 15 MAR 2O30 (with typo)
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{2})',  # 20/07/24
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{2})',  # 20 MAY 24
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024-07-20
    r'(\d{4}[A-Za-z]{3,}\d{2})',  # 2024MAY20
    r'(\d{2}[A-Za-z]{3,}\d{4})',  # 20MAY2024
    r'(?:DX3\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*[A-Za-z]{3,}\s*(\d{4}|\d{2})))',
    r'(?:exp\.?\s*date\s*[:\-]?\s*(\d{2}\s*\d{2}\s*\d{4}))',  # Exp. Date: 20 05 2025
    r'(\d{4}[A-Za-z]{3}\d{2})',  # 2025MAY11
    r'(?:best\s*before\s*[:\-]?\s*(\d+)\s*(days?|months?|years?))',  # Best Before: 6 months
    r'(?:best\s*before\s*[:\-]?\s*(three)\s*(months?))',
    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*\d{4})',
    r'\bUSE BY\s+(\d{1,2}[A-Za-z]{3}\d{4})\b',
    r"Exp\.Date\s*(\d{2}[A-Z]{3}\d{4})",
    r"EXP:\d{4}/\d{2}/\d{4}/\d{1}/[A-Z]"
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]  # Return the first match
    return "No expiry date found"

def perform_ocr_on_image(image_path):
    img = cv2.imread(image_path)
    # Convert PIL image to OpenCV format (PaddleOCR works with OpenCV images)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Perform OCR on the image
    ocr_results = ocr.ocr(img_cv)

    # Extract text from the OCR results
    text = ""
    for line in ocr_results:
        for word_info in line:
            text += word_info[1][0] + " "
    expiry_date = extract_expiry_date(text)
    return text, expiry_date

# HTML and CSS Styling


HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified App</title>
    <style>
        /* Reset default styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            background-color: #f4f4f4;
        }

        .background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('robot1.jpg') no-repeat center center/cover;
            animation: slideShow 30s infinite;
            z-index: -1;
        }

        @keyframes slideShow {
            0% { background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkPqMo2Vv7AvThKBEEYKScumbcu5ZngfsDGg&s') no-repeat center center/cover; }
            33% { background: url('https://media-cldnry.s-nbcnews.com/image/upload/t_social_share_1024x768_scale,f_auto,q_auto:best/newscms/2023_11/3598896/230317-derm-skin-care-routine-bd-2x1.jpg') no-repeat center center/cover; }
            66% { background: url('https://media.istockphoto.com/id/1184804468/photo/industrial-technology-concept-factory-automation-smart-factory-industry-4-0.jpg?s=612x612&w=0&k=20&c=1MaCUFJnqZmuugNhMyL5kt4q0BMwiNpzmnJbSggBE6I=') no-repeat center center/cover; }
            100% { background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkPqMo2Vv7AvThKBEEYKScumbcu5ZngfsDGg&s') no-repeat center center/cover; }
        }

        .container {
            position: relative;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 60px;
            width: 100%;
            max-width: 1200px;
            height: auto;
            min-height: 500px;
            border-radius: 20px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 30px;
            color: #333;
        }

        .button-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }

        button {
            padding: 15px 30px;
            font-size: 20px;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
            width: 300px;
        }

        button:hover {
            background-color: #45a049;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2em;
            }

            .container {
                padding: 40px;
            }

            button {
                width: 250px;
                font-size: 18px;
            }
        }

        @media (max-width: 480px) {
            h1 {
                font-size: 1.5em;
            }

            .container {
                padding: 20px;
            }

            button {
                width: 200px;
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="background"></div>
    <div class="container">
        <h1>Flipkart Robotics Services</h1>
        <div class="button-container">
            <button onclick="location.href='/detect_objects'">Object Detection</button>
            <button onclick="location.href='/text_extraction'">Text Extraction And Expiry Date</button>
            <button onclick="location.href='/freshness_detection'">Freshness Detection</button>
        </div>
    </div>
</body>
</html>
"""

UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            overflow: auto;
            background: linear-gradient(45deg, #d3d3d3, #87ceeb, #d3d3d3);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
        }

        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            position: relative;
            background-color: white;
            padding: 20px;
            max-width: 90%;
            width: 600px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            z-index: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            max-height: 90vh; /* Limit the height */
            overflow-y: auto; /* Enable scrolling */
        }

        h1 {
            font-size: 2.5em; /* Adjusted text size */
            margin-bottom: 30px;
        }

        input[type="file"] {
            font-size: 1.2em; /* Adjusted text size */
            margin-bottom: 20px;
            display: block;
        }

        button {
            padding: 20px 35px;
            font-size: 1.5em;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #45a049;
        }

        img {
            max-width: 100%;
            height: auto;
            margin-top: 20px;
            max-height: 400px; /* Limit image height */
        }

        #output {
            margin-top: 20px;
            white-space: pre-wrap;
            font-size: 1.2em; /* Adjusted text size */
            max-width: 100%;
            word-wrap: break-word;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2em; /* Adjusted for smaller screens */
            }

            input[type="file"] {
                font-size: 1.1em;
            }

            button {
                font-size: 1.3em;
                padding: 15px 30px;
            }

            .container {
                padding: 20px;
                width: 90%;
                max-width: 100%;
            }

            #output {
                font-size: 1.1em;
            }
        }

        @media (max-width: 480px) {
            h1 {
                font-size: 1.8em; /* Further adjustment for smaller screens */
            }

            input[type="file"] {
                font-size: 1em;
            }

            button {
                font-size: 1.1em;
                padding: 12px 25px;
            }

            .container {
                padding: 10px;
                width: 100%;
            }

            #output {
                font-size: 1em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Upload and Process</button>
        </form>
        {% if image_url %}
        <img id="uploaded-image" src="{{ image_url }}" alt="Uploaded Image">
        {% endif %}
        {% if result %}
        <div id="output">{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

# Routes
@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/detect_objects', methods=['GET', 'POST'])
def detect_objects():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the image using OpenCV
        img = cv2.imread(filepath)

        # Perform object detection
        results = yolo_model(img)

        # Render the results on the image
        results.render()  # Draws bounding boxes on the image

        # Save the rendered image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
        cv2.imwrite(result_path, results.ims[0])  # Corrected to use 'ims'

        return render_template_string(
            UPLOAD_HTML,
            title="Object Detection",
            result=f"Detected {len(results.xyxy[0])} objects",
            image_url=f'/uploads/result_{filename}'
        )

    return render_template_string(UPLOAD_HTML, title="Object Detection")

@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_text, expiry_date = perform_ocr_on_image(filepath)
        result_message = f"Extracted Text: {extracted_text}<br>Expiry Date: {expiry_date}"
        return render_template_string(UPLOAD_HTML, title="Text Extraction", result=result_message, image_url=f'/uploads/{filename}')
    return render_template_string(UPLOAD_HTML, title="Text Extraction")

@app.route('/freshness_detection', methods=['GET', 'POST'])
def freshness_detection():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        prediction = tensorflow_model.predict(img)
        class_index = np.argmax(prediction, axis=-1)[0]
        class_name = class_names.get(class_index, "Unknown")

        return render_template_string(UPLOAD_HTML, title="Freshness Detection", result=f"Class: {class_name}", image_url=f'/uploads/{filename}')
    return render_template_string(UPLOAD_HTML, title="Freshness Detection")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prevent browser caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

# Entry point
if __name__ == '__main__':
    app.run(debug=True)
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Add logs to the routes
@app.route('/')
def home():
    logger.info('Home page accessed')
    return render_template_string(HOME_HTML)

@app.route('/detect_objects', methods=['GET', 'POST'])
def detect_objects():
    logger.info('Object detection page accessed')
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            logger.error('No selected file')
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the image using OpenCV
        img = cv2.imread(filepath)

        # Perform object detection
        results = yolo_model(img)

        # Render the results on the image
        results.render()  # Draws bounding boxes on the image

        # Save the rendered image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
        cv2.imwrite(result_path, results.ims[0])  # Corrected to use 'ims'

        logger.info('Object detection completed')
        return render_template_string(
            UPLOAD_HTML,
            title="Object Detection",
            result=f"Detected {len(results.xyxy[0])} objects",
            image_url=f'/uploads/result_{filename}'
        )

    return render_template_string(UPLOAD_HTML, title="Object Detection")

@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    logger.info('Text extraction page accessed')
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            logger.error('No selected file')
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_text, expiry_date = perform_ocr_on_image(filepath)
        result_message = f"Extracted Text: {extracted_text}<br>Expiry Date: {expiry_date}"
        logger.info('Text extraction completed')
        return render_template_string(UPLOAD_HTML, title="Text Extraction", result=result_message, image_url=f'/uploads/{filename}')
    return render_template_string(UPLOAD_HTML, title="Text Extraction")

@app.route('/freshness_detection', methods=['GET', 'POST'])
def freshness_detection():
    logger.info('Freshness detection page accessed')
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            logger.error('No selected file')
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        prediction = tensorflow_model.predict(img)
        class_index = np.argmax(prediction, axis=-1)[0]
        class_name = class_names.get(class_index, "Unknown")

        logger.info('Freshness detection completed')
        return render_template_string(UPLOAD_HTML, title="Freshness Detection", result=f"Class: {class_name}", image_url=f'/uploads/{filename}')
    return render_template_string(UPLOAD_HTML, title="Freshness Detection")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logger.info('File uploaded')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prevent browser caching
@app.after_request
def add_header(response):
    logger.info('Response sent')
    response.headers['Cache-Control'] = 'no-store'
    return response