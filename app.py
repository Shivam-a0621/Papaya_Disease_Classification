from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and labels
model = load_model('keras_model.h5', compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image to 224x224 and crop from center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Create the input array for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the uploaded image
            preprocessed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(preprocessed_image)
            index = np.argmax(prediction)
            class_name = class_names[index][2:].strip()  # Remove the label number and newline
            confidence_score = prediction[0][index]

            prediction_text = f'Prediction: {class_name} (Confidence Score: {confidence_score:.2f})'

            return render_template('index.html', prediction=prediction_text, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
