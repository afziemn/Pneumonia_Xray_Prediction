
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "C:/Users/Afzi/Documents/BINUS/Projects/Parkinson_Xray_Prediction/model.h5"  # Update with the actual path to your trained model
IMG_HEIGHT, IMG_WIDTH = 264, 64

# Load model
model = load_model(MODEL_PATH)

# Define labels
labels = {0: 'Normal', 1: 'Pneumonia'}

# Streamlit app
st.title("Pneumonia Detection from X-Ray")
st.write("Upload an X-Ray image to predict if the patient has pneumonia or not.")

# Image uploader
uploaded_file = st.file_uploader("Upload an X-Ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    # Preprocess the image
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize image
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict
    st.write("Predicting...")
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    
    # Display result
    st.write(f"Prediction: **{labels[predicted_class]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")