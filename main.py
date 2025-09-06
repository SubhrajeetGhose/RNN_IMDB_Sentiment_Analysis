# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 2: Load the pre-trained model
# Prefer the new `.keras` format, fallback to `.h5`
MODEL_PATH_KERAS = "simple_rnn_imdb.keras"
MODEL_PATH_H5 = "simple_rnn_imdb.h5"

if os.path.exists(MODEL_PATH_KERAS):
    model = load_model(MODEL_PATH_KERAS, compile=False)
elif os.path.exists(MODEL_PATH_H5):
    model = load_model(MODEL_PATH_H5, compile=False)
else:
    st.error("No model file found. Please upload simple_rnn_imdb.keras or simple_rnn_imdb.h5")
    st.stop()

# Step 3: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 4: Streamlit App
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as **positive** or **negative**.')

# User input
user_input = st.text_area('✍️ Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before classification.")
    else:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = '😊 Positive' if prediction[0][0] > 0.5 else '😞 Negative'

        # Display the result
        st.success(f'Sentiment: {sentiment}')
        st.info(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('👆 Please enter a movie review above and press *Classify*.')
