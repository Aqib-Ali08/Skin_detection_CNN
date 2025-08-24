from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import io
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved model
MODEL_PATH = "C:\\Users\\Chikpea\\Desktop\\Clg Projects\\Skin_detection_CNN\\backend\\Machine Learning\\trained_model_2.keras" 

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

# Define class names (ensure this matches your model's training setup)
CLASS_NAMES = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
    'squamous cell carcinoma', 'vascular lesion'
]  # Replace with your actual class names

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict the class of an uploaded image."""
    try:
        logger.info("Prediction request received.")
        if 'file' not in request.files:
            logger.warning("No file part in the request.")
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected for upload.")
            return jsonify({'error': 'No file selected for upload'}), 400
        img = io.BytesIO(file.read())
        image = load_img(img, target_size=(180,180,3))
        image_batch = np.expand_dims(image, axis=0) 
        predictions = model.predict(image_batch)
        predicted_class_index = int(np.argmax(predictions))
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        logger.info(f"Predicted class: {predicted_class_name}")
        return jsonify({
            'predicted_class_index': predicted_class_index,
            'predicted_class_name': predicted_class_name,
            'confidence_scores': predictions.tolist()[0]
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
