from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/RC PRASAD/Desktop/python ws/project/ad_deploy.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))  # Resize image to your model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image_bytes = file.read()
    
    img_array = preprocess_image(image_bytes)
    
    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    class_name = 'not a advertisement' if class_index == 0 else 'advertisement'
    probability = float(predictions[0][class_index])
    
    return jsonify({'class_name': class_name, 'probability': probability})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

