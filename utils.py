import numpy as np
from PIL import Image
import tensorflow as tf
import os

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_model(model_path):
    """Load the trained model"""
    return tf.keras.models.load_model(model_path)

def get_class_names():
    """Get class names from the dataset directory"""
    # Updated class names to match the exact folder names
    return ['roses', 'sunflowers', 'tulips']

def predict_image(model, image_path):
    """Make prediction on a single image"""
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    class_names = get_class_names()
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence 