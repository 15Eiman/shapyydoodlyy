import os
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import joblib
import math

def create_circle(size):
    image = Image.new('L', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    padding = size // 8
    draw.ellipse([padding, padding, size - padding, size - padding], fill='black')
    return np.array(image)

def create_square(size):
    image = Image.new('L', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    padding = size // 8
    draw.rectangle([padding, padding, size - padding, size - padding], fill='black')
    return np.array(image)

def create_triangle(size):
    image = Image.new('L', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    padding = size // 8
    points = [
        (size // 2, padding),
        (padding, size - padding),
        (size - padding, size - padding)
    ]
    draw.polygon(points, fill='black')
    return np.array(image)

def create_star(size):
    image = Image.new('L', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    padding = size // 8
    center = size // 2
    outer_radius = (size - 2 * padding) // 2
    inner_radius = outer_radius // 2
    points = []
    
    for i in range(10):
        angle = math.pi / 2 + (2 * math.pi * i) / 10
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        points.append((x, y))
    
    draw.polygon(points, fill='black')
    return np.array(image)

def extract_features(image):
    # Extract HOG features
    features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), feature_vector=True)
    return features

def create_dataset(size=64, samples_per_shape=100):
    X = []
    y = []
    shape_generators = {
        'circle': create_circle,
        'square': create_square,
        'triangle': create_triangle,
        'star': create_star
    }
    
    for shape_name, shape_func in shape_generators.items():
        print(f"Generating {shape_name} samples...")
        for _ in range(samples_per_shape):
            # Create base shape
            image = shape_func(size)
            
            # Extract features
            features = extract_features(image)
            
            X.append(features)
            y.append(shape_name)
            
            # Save one example of each shape
            if _ == 0:
                Image.fromarray(image).save(f'static/example_{shape_name}.png')
    
    return np.array(X), np.array(y)

def train_model():
    print("Creating training dataset...")
    X, y = create_dataset(samples_per_shape=100)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Saving model...")
    joblib.dump(model, 'shapes_model.joblib')
    print("Model saved as shapes_model.joblib")

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    train_model() 