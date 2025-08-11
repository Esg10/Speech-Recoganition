import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
import keras
import joblib
from utils import get_features

st.set_page_config(page_title="Speech Emotion Recognition (Local)", layout="centered")

# Setup
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

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
pca = load_pca()

# Functions
def predict_emotion(file_path):
    emotion, avg_probs, labels = get_features(
        file_path,
        pca=pca,
        scaler=scaler,
        get_probs=True,
        model=model,
        encoder=encoder
    )
    return emotion, avg_probs, labels

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-Spectrogram')
    st.pyplot(fig)

def record_audio(filename="recorded.wav", duration=3, fs=16000):
    st.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    return filename

# UI
st.title("🎤 Speech Emotion Recognition (Local)")

option = st.radio("Choose an input method:", ("Upload Audio File", "Record Audio"))

file_path = None

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("📤 Upload .wav File", type=["wav"])
    if uploaded_file:
        path = os.path.join(DATA_DIR, "uploaded.wav")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        file_path = path

elif option == "Record Audio":
    if st.button("🎙️ Record 3 Seconds"):
        file_path = os.path.join(DATA_DIR, "recorded.wav")
        record_audio(file_path)

# Prediction
if file_path:
    st.audio(file_path, format="audio/wav")
    plot_spectrogram(file_path)

    if st.button("🚀 Predict Emotion"):
        emotion, probs, labels = predict_emotion(file_path)
        st.success(f"🎯 **Predicted Emotion:** `{emotion}`")

        # Bar plot
        fig, ax = plt.subplots()
        bars = ax.barh(labels, probs, color="skyblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Emotion Confidence")
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{prob:.2f}", va="center")
        st.pyplot(fig)

if __name__ == "__main__":
    pass
