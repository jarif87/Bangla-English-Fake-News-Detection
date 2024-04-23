import gradio as gr
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the dataset
df = pd.read_csv("bangla_english_fake.csv")  # Update the filename with your dataset

# Preprocess the data
max_len = 1600
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
text_seq = tokenizer.texts_to_sequences(df["text"])

# Load the LSTM model
model = load_model("bangla_english_fake_model.h5")

# Define function for fake news detection
def classify_fake_news(text):
    # Tokenize and pad the input text sequence
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len)
    # Predict the label for the input text
    prediction = model.predict(padded_sequence)[0]
    # Convert prediction to label
    label = "Fake" if prediction < 0.5 else "Real"
    return label

# Define Gradio interface
iface = gr.Interface(
    fn=classify_fake_news,
    inputs="text",
    outputs="label",
    title="Bangla & English Fake News Detection",
    description="Enter a news article to detect if it's fake or real."
)

# Launch the Gradio interface
iface.launch(share=True)
