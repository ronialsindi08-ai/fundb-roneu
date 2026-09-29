import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- KONFIGURATION (Hier Dateiname angepasst) ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_FILE = "fundbuero_data.json"

@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Fehler beim Laden des Modells: {e}")
            return None
    return None

@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            # Entfernt eventuelle Teachable-Machine Präfixe wie "0 " oder "1 "
            labels_list = []
            for line in f.readlines():
                cleaned = line.strip()
                if cleaned:
                    # Falls Labels als "0 Flasche" formatiert sind, Präfix abschneiden
                    parts = cleaned.split(" ", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        labels_list.append(parts[1])
                    else:
                        labels_list.append(cleaned)
            return labels_list
    return []

model = load_keras_model()
labels = load_labels()

# KI-Vorhersage Funktion
def predict_image(image):
    if model is None:
        return "Fehler: 'keras_model.h5' nicht gefunden!"
    if not labels:
        return "Fehler: 'labels.txt' nicht gefunden oder leer!"
    
    # Bildgröße anpassen (Standard für Teachable Machine / MobileNet ist 224x224)
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    
    # Normalisierung
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx]) * 100
    
    if class_idx < len(labels):
        return f"{labels[class_idx]} ({confidence:.1f}% Sicherheit)"
    return "Unbekannte Kategorie"
