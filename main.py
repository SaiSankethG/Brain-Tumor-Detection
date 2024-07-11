import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
import io

# Load the model from the pickle file
model = None
try:
    model = load_model('TumorDetection (1).h5')
except Exception as e:
    st.error(f"Error loading model: {e}")


def img_pred(uploaded_image):
    try:
        img = Image.open(uploaded_image)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvImage, (150, 150))
        img = img.reshape(1, 150, 150, 3)
        p = model.predict(img)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            prediction = 'Glioma Tumor'
        elif p == 1:
            prediction = 'No Tumor'
        elif p == 2:
            prediction = 'Meningioma Tumor'
        else:
            prediction = 'Pituitary Tumor'

        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


# Streamlit app
st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Process and predict
    if st.button('Predict'):
        if model is not None:
            prediction = img_pred(uploaded_file)
            if prediction:
                st.write(f"The Model predicts that it is a: {prediction}")
        else:
            st.error("Model is not loaded correctly.")

