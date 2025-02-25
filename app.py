import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
pip install flask tensorflow pillow numpy

# Load trained model
MODEL_PATH = "rock_paper_scissors_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Flask app
app = Flask(__name__)

# Define class labels (must match your training classes)
CLASS_NAMES = ['rock', 'paper', 'scissors']

# Function to preprocess images
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# API route to receive and predict images
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return jsonify({"gesture": predicted_class})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
