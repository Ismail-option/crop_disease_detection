from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load model
print("Loading model...")
model = load_model('crop_disease_model.h5')
print("✅ Model loaded!")

# Class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
               'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
               'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Treatment recommendations
treatments = {
    'Apple___Apple_scab': 'Apply fungicide sprays containing captan or myclobutanil. Remove and destroy infected leaves.',
    'Apple___Black_rot': 'Prune infected branches. Apply copper-based fungicides during the growing season.',
    'Apple___Cedar_apple_rust': 'Apply fungicides containing myclobutanil or propiconazole. Remove nearby cedar trees if possible.',
    'Apple___healthy': 'Your apple plant is healthy! Maintain regular watering and fertilization.',
    'Blueberry___healthy': 'Your blueberry plant is healthy! Maintain proper soil pH between 4.5 and 5.5.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur-based fungicides. Ensure good air circulation around plants.',
    'Cherry_(including_sour)___healthy': 'Your cherry plant is healthy! Maintain regular pruning and watering.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply foliar fungicides. Practice crop rotation and remove infected debris.',
    'Corn_(maize)___Common_rust_': 'Apply fungicides containing azoxystrobin. Plant rust-resistant maize varieties.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties. Apply fungicides at early stages of infection.',
    'Corn_(maize)___healthy': 'Your maize plant is healthy! Maintain adequate fertilization and irrigation.',
    'Grape___Black_rot': 'Apply fungicides containing myclobutanil. Remove mummified berries and infected leaves.',
    'Grape___Esca_(Black_Measles)': 'Prune infected wood. Apply fungicides and avoid wounding vines unnecessarily.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply copper-based fungicides. Remove and destroy infected plant material.',
    'Grape___healthy': 'Your grape plant is healthy! Maintain proper pruning and irrigation practices.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure exists. Remove infected trees to prevent spread. Control psyllid insects.',
    'Peach___Bacterial_spot': 'Apply copper-based bactericides. Avoid overhead irrigation and plant resistant varieties.',
    'Peach___healthy': 'Your peach plant is healthy! Maintain regular fertilization and pest monitoring.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides. Use disease-free seeds and avoid overhead watering.',
    'Pepper,_bell___healthy': 'Your pepper plant is healthy! Maintain adequate watering and nutrient supply.',
    'Potato___Early_blight': 'Apply fungicides containing chlorothalonil. Practice crop rotation and remove infected debris.',
    'Potato___Late_blight': 'Apply fungicides immediately. Remove and destroy infected plants to prevent spread.',
    'Potato___healthy': 'Your potato plant is healthy! Maintain proper hilling and irrigation practices.',
    'Raspberry___healthy': 'Your raspberry plant is healthy! Maintain proper pruning and soil drainage.',
    'Soybean___healthy': 'Your soybean plant is healthy! Maintain adequate fertilization and weed control.',
    'Squash___Powdery_mildew': 'Apply sulfur or potassium bicarbonate fungicides. Ensure good air circulation.',
    'Strawberry___Leaf_scorch': 'Apply fungicides containing captan. Remove infected leaves and avoid overhead irrigation.',
    'Strawberry___healthy': 'Your strawberry plant is healthy! Maintain proper mulching and irrigation.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Use disease-free seeds and practice crop rotation.',
    'Tomato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Remove lower infected leaves.',
    'Tomato___Late_blight': 'Apply fungicides immediately. Remove and destroy all infected plant material.',
    'Tomato___Leaf_Mold': 'Improve air circulation. Apply fungicides containing chlorothalonil.',
    'Tomato___Septoria_leaf_spot': 'Apply fungicides containing mancozeb. Remove infected leaves immediately.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides or neem oil. Increase humidity around plants.',
    'Tomato___Target_Spot': 'Apply fungicides containing azoxystrobin. Practice crop rotation.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure exists. Remove infected plants. Control whitefly population.',
    'Tomato___Tomato_mosaic_virus': 'No cure exists. Remove and destroy infected plants. Disinfect tools regularly.',
    'Tomato___healthy': 'Your tomato plant is healthy! Maintain regular watering and fertilization.'
}

def predict_disease(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    disease = class_names[predicted_class]
    treatment = treatments.get(disease, 'Consult an agricultural expert for treatment advice.')
    return disease, confidence, treatment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file temporarily
    upload_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(upload_path)
    
    # Make prediction
    disease, confidence, treatment = predict_disease(upload_path)
    
    # Clean up
    os.remove(upload_path)
    
    return jsonify({
        'disease': disease,
        'confidence': f'{confidence:.2f}%',
        'treatment': treatment
    })

if __name__ == '__main__':
    app.run(debug=True)