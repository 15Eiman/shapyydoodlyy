from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import os
import joblib

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'shapes_model.joblib')
model = joblib.load(model_path)
img_size = 64
target_features = 1568  # Number of features expected by the model

def preprocess_image(image_data):
    # Decode base64 image
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors (we trained on black shapes on white background)
    image = ImageOps.invert(image)
    
    # Resize to match model input size
    image = image.resize((img_size, img_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Downsample to match the expected number of features
    # Calculate the dimensions needed to get target_features
    downsample_size = int(np.sqrt(target_features/4))  # Divide by 4 to account for 2x2 blocks
    image = Image.fromarray((img_array * 255).astype(np.uint8))
    image = image.resize((downsample_size * 2, downsample_size * 2))
    
    # Convert back to numpy and normalize
    img_array = np.array(image) / 255.0
    
    # Create blocks of 2x2 pixels and calculate their average
    blocks = img_array.reshape(downsample_size, 2, downsample_size, 2).mean(axis=(1, 3))
    
    # Flatten and ensure we have exactly target_features
    features = blocks.reshape(-1)
    if len(features) < target_features:
        features = np.pad(features, (0, target_features - len(features)))
    elif len(features) > target_features:
        features = features[:target_features]
        
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Preprocess image and extract features
        features = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get confidence scores for all classes
        confidence_scores = {
            class_name: float(prob)
            for class_name, prob in zip(model.classes_, probabilities)
        }
        
        return jsonify({
            'prediction': prediction,
            'confidence_scores': confidence_scores
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True)
else:
    # For Vercel
    app = app 