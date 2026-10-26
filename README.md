# Crop Disease Detection System 🌱🤖

An AI-powered crop disease detection system built using Deep Learning and Computer Vision. The system uses a trained neural network model to classify plant diseases from leaf images and provide predictions through a web interface.

## Project Overview

Crop diseases can significantly reduce agricultural productivity. This project aims to assist farmers and agricultural professionals by using Artificial Intelligence to detect diseases from plant leaf images.

The application allows users to upload an image of a crop leaf, and the AI model analyzes the image to predict the possible disease.

## Features

- 🌱 Upload crop leaf images
- 🤖 Deep learning-based disease classification
- 📷 Image processing and prediction
- 🌐 Web interface built with Flask
- 📊 Model training and evaluation visualization

## Technologies Used

- Python
- TensorFlow / Keras
- Deep Learning
- Convolutional Neural Networks (CNN)
- Computer Vision
- Flask
- HTML/CSS
- NumPy
- Matplotlib

## Project Structure

crop_disease_detection/

├── app.py
├── model.py
├── retrain.py
├── templates/
│   └── index.html
├── static/
│   ├── background.jpg
│   └── background2.jpg
├── training_results.png
├── requirements.txt
└── README.md


## Dataset

The model was trained using a crop disease image dataset obtained from Kaggle.

The dataset contains thousands of labeled plant leaf images used for training and evaluating the deep learning model.

## How to Run the Project

### 1. Clone the repository
git clone https://github.com/yourusername/crop_disease_detection.git

### 2. Navigate into the project folder

cd crop_disease_detection

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the virtual environment

Windows:

venv\Scripts\activate

### 5. Install dependencies

pip install -r requirements.txt

### 6. Run the application

python app.py

Open your browser and visit:

http://127.0.0.1:5000

## Model

The project uses a deep learning image classification model trained to identify crop diseases from leaf images.

The trained model file is not included in this repository due to its large size.

## Future Improvements

- Improve model accuracy
- Reduce model size for mobile deployment
- Add more crop disease classes
- Deploy the application online
- Integrate real-time camera detection

## Author

Abdulrahman Ismail

GitHub:
https://github.com/yourusername