import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

activities = ["Garbage Classifier"]
choice = st.sidebar.selectbox("Select", activities)


def import_and_predict(image_data, model):

    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]
    prediction = model.predict_classes(img_reshape)

    return prediction


if choice == "Garbage Classifier":

    garbageClasses = ['biological', 'cardboard',
                      'glass', 'metal', 'paper', 'plastic', 'trash']
    garbageModel = load_model('GarbageClassifier.h5')

    st.title("Welcome to Garbage CLassifier")
    st.header("Identify what's the Garbage!")

    if st.checkbox("These are the classes of Garbage it can identify"):
        st.write(garbageClasses)

    file = st.file_uploader("Please upload a Garbage Image", type=[
                            "jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, garbageModel)
        result = predictions
        st.write(predictions)

        if result == 0:
            st.write("Prediction : biological")
        elif result == 1:
            st.write("Prediction : cardboard")
        elif result == 2:
            st.write("Prediction : glass")
        elif result == 3:
            st.write("Prediction : metal")
        elif result == 4:
            st.write("Prediction : paper")
        elif result == 5:
            st.write("Prediction : plastic")
        elif result == 6:
            st.write("Prediction : trash")
