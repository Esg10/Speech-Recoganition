import numpy as np
import librosa
import torch
import joblib
import streamlit as st

# Augmentation Functions

def noise(data, noise_level=0.035):
    noise_amp = noise_level * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    data = data.astype(np.float32)
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sample_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=pitch_factor)

def shift(data, shift_max=0.2):
    shift_amt = int(np.random.uniform(-shift_max, shift_max) * len(data))
    return np.roll(data, shift_amt)

def volume(data, gain_min=0.8, gain_max=1.2):
    gain = np.random.uniform(gain_min, gain_max)
    return data * gain

# Handcrafted Feature Extraction

def extract_features(data, sample_rate):
    result = np.array([])

    # Temporal Features
    zcr = librosa.feature.zero_crossing_rate(y=data)[0]
    rms = librosa.feature.rms(y=data)[0]
    result = np.hstack((result, np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)))

    # Frequency Features
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)

    result = np.hstack((result,
                        np.mean(chroma_stft, axis=1),
                        np.mean(mfcc, axis=1),
                        np.std(mfcc, axis=1),
                        np.mean(delta_mfcc, axis=1),
                        np.mean(mel, axis=1)[:20]))

    # Spectral Features
    centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)[0]
    bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)[0]
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)

    result = np.hstack((result,
                        np.mean(centroid), np.std(centroid),
                        np.mean(bw), np.std(bw),
                        np.mean(rolloff), np.std(rolloff),
                        np.mean(contrast, axis=1)))

    # Tonal Features
    y_harmonic = librosa.effects.harmonic(data)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate)
    result = np.hstack((result, np.mean(tonnetz, axis=1)))

    return result

# Wav2Vec2 Components (Lazy Load)
@st.cache_resource
def load_wav2vec():
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        return processor, model
    except Exception as e:
        st.error(f"❌ Failed to load Wav2Vec2 model: {e}")
        raise

# Final Feature Getter
def get_features(path, emotion=None, pca=None, scaler=None, get_probs=False, model=None, encoder=None):
    try:
        data, sr = librosa.load(path, duration=2.5, offset=0.6, sr=None)
    except Exception as e:
        st.error(f"❌ Error loading audio file: {e}")
        raise

    if sr != 16000:
        try:
            data_wav = librosa.resample(data, orig_sr=sr, target_sr=16000)
        except Exception as e:
            st.error(f"❌ Error during resampling: {e}")
            raise
    else:
        data_wav = data

    # Load Wav2Vec2
    try:
        processor, model_wav2vec = load_wav2vec()
        with torch.no_grad():
            input_values = processor(data_wav, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            w2v = model_wav2vec(input_values).last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        st.error(f"❌ Error extracting Wav2Vec2 embedding: {e}")
        raise

    # Define augmentations 
    augmentations = [
        lambda x: x,
        noise,
        shift,
        lambda x: pitch(x, sr)
    ]

    features = []
    for aug in augmentations:
        try:
            data_aug = aug(data)
            handcrafted = extract_features(data_aug, sr)
            reduced = pca.transform([handcrafted])[0] if pca else handcrafted
            combined = np.hstack((reduced, w2v))
            combined = scaler.transform([combined])[0] if scaler else combined
            features.append(combined)
        except Exception as e:
            st.warning(f"⚠️ Augmentation `{aug.__name__}` failed: {e}")

    features = np.array(features)

    # Predict if requested
    if get_probs and model is not None and encoder is not None:
        try:
            probs = np.array([model.predict(np.expand_dims(f, axis=0))[0] for f in features])
            avg_probs = probs[0] * 0.5 + np.sum(probs[1:], axis=0) * (0.5 / (len(probs) - 1))
            predicted_label = encoder.categories_[0][np.argmax(avg_probs)]
            return predicted_label, avg_probs, encoder.categories_[0]
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            return "error", np.zeros(len(encoder.categories_[0])), encoder.categories_[0]

    return features
