from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = load_model('mobilenet_model.h5')

class_names = ['Abdeen Palace Museum', 'Abu Simbel', 'Mortuary temple of Hatshepsut', 'Qasr El Nil Bridge', 'The Giza Pyramids']

def prepare_image(img, target_size):
    # Preprocess the image as per MobileNetV2 requirements
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale as per the training phase
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was properly uploaded to the endpoint
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Retrieve the image file from the request
    file = request.files['file']

    # Open the image file and convert it to a format suitable for the model
    img = Image.open(io.BytesIO(file.read()))
    processed_image = prepare_image(img, target_size=(224, 224))

    # Make a prediction using the loaded model
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Get the class label
    class_label = class_names[predicted_class[0]]
    confidence = np.max(predictions)  

    # Return the result as a JSON response
    return jsonify({'class': class_label, 'confidence': float(confidence)})

if __name__=='__main__':
    
    api.run(
        host= '127.0.0.1',
        port= 5000,
        debug= True
    )
