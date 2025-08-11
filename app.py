import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import librosa.display
import keras
from utils import get_features  

st.set_page_config(page_title="Speech Emotion Recognition 🎙️", layout="centered")

# Setup
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache model and tools
@st.cache_resource
def load_model():
    return keras.models.load_model("emotion_model.keras")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("standard_scaler.pkl")

@st.cache_resource
def load_pca():
    return joblib.load("pca_handcrafted.pkl")

model = load_model()
encoder = load_encoder()
scaler = load_scaler()
pca_handcrafted = load_pca()

# Predict function
def predict_emotion(file_path):
    try:
        emotion, avg_probs, labels = get_features(
            file_path,
            pca=pca_handcrafted,
            scaler=scaler,
            get_probs=True,
            model=model,
            encoder=encoder
        )
        return emotion, avg_probs, labels
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return "error", [], []

# Plot function
def plot_probabilities(probabilities, labels):
    fig, ax = plt.subplots()
    bars = ax.barh(labels, probabilities, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Prediction Confidence")
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{prob:.2f}", va='center')
    st.pyplot(fig)

# UI
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file to detect the emotion in speech.")

if 'file_path' not in st.session_state:
    st.session_state.file_path = None

uploaded_file = st.file_uploader("📤 Upload Audio File (.wav)", type=["wav"])
if uploaded_file:
    path = os.path.join(DATA_DIR, "uploaded.wav")
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.file_path = path

# Predict UI
if st.session_state.file_path:
    st.audio(st.session_state.file_path, format="audio/wav")

    if st.button("🚀 Predict Emotion"):
        with st.spinner("Analyzing emotion..."):
            emotion, probs, labels = predict_emotion(st.session_state.file_path)
        if emotion != "error":
            st.success(f"🎯 **Predicted Emotion:** `{emotion}`")
            plot_probabilities(probs, labels)
