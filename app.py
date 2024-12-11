import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model("simple_rnn_model.h5")

def decode_review(text):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in text])
  
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=10000)
  return padded_review

def predict_sentiment(text):
  padded_review = preprocess_text(text)
  prediction = model.predict(padded_review)[0][0]
  sentiment = 'positive' if prediction > 0.5 else 'negative'
  return sentiment, prediction

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Movie review")

if st.button("Classify"):
   sentiment, prediction = predict_sentiment(user_input)
   st.write("Sentiment : ",sentiment)
   st.write("prediction score : ",prediction)
else:
   st.write("Please enter a review")