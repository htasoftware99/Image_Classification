import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

st.header('Image Classification Model')
model = load_model("Image_Classifying.keras")

data_category = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height = 180
img_width = 180

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = image.resize((img_height, img_width))
    
    # Convert the image to an array
    img_array = np.array(image)
    
    # Preprocess the image
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Display the image and the prediction
    st.image(image, caption='Uploaded Image', width=200)
    st.write('Veg/Fruit in image is ' + data_category[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score)*100))
