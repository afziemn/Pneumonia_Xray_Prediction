# Pneumonia X-ray Prediction

This project is a web application that predicts whether a patient has pneumonia based on X-ray images. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and is deployed using Streamlit.

## Features
- **Interactive Web App**: Upload X-ray images for real-time predictions.
- **Deep Learning Model**: Classifies images as `Normal` or `Pneumonia`.
- **Confidence Score**: Provides the model's confidence in its predictions.

## Folder Structure
Pneumonia_Xray_Prediction/ 
├── pneumonia_normal_xray.zip # Dataset 
├── app.py # Streamlit app 
├── Pneumonia_Xray_Prediction.ipynb # Jupyter Notebook 
├── model.h5 # Trained model

## Dataset
Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1wRxA7Kqug_3F6_-RABcHII3YjAnWeCpY?usp=sharing) and place it in the root directory of the project.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/afziemn/Pneumonia_Xray_Prediction.git
   cd Pneumonia_Xray_Prediction

2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py

## Model
The CNN model was trained on a dataset of labeled X-ray images (Normal/Pneumonia). The architecture is designed for binary classification and was saved as model.h5.
