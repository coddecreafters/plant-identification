from flask import Flask, request, render_template, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from utils import load_model, predict_image
from train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load or train the model
try:
    logger.info("Attempting to load existing model...")
    model = load_model('plant_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.info("Training new model...")
    try:
        train_model()
        model = load_model('plant_model.h5')
        logger.info("New model trained and loaded successfully")
    except Exception as e:
        logger.error(f"Error training new model: {str(e)}")
        raise

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                logger.info(f"File saved successfully at {filepath}")
                
                predicted_class, confidence = predict_image(model, filepath)
                logger.info(f"Prediction successful: {predicted_class} with confidence {confidence}")
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'prediction': predicted_class,
                    'confidence': f'{confidence:.2%}'
                })
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
