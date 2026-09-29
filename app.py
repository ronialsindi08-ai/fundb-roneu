import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- KONFIGURATION & MODELL LADEN ---
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
DATA_FILE = "fundbuero_data.json"

@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    return []

model = load_keras_model()
labels = load_labels()

# --- DATEN-HANDLING ---
def load_items():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_items(items):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# KI-Vorhersage Funktion
def predict_image(image):
    if model is None or not labels:
        return "Unbekannt (Modell/Labels fehlen)"
    
    # Bild für Keras vorbereiten (Standard 224x224, ggf. an dein Modell anpassen)
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    if class_idx < len(labels):
        return labels[class_idx]
    return "Unbekannt"

# --- STREAMLIT UI ---
st.set_page_config(page_title="KI Fundbüro", page_icon="🔍")
st.title("🔍 KI-Fundbüro")

tab1, tab2 = st.tabs(["🔎 Gegenstand suchen", "➕ Fundstück melden (mit KI)"])

# TAB 1: SUCHEN
with tab1:
    st.header("Fundstücke durchsuchen")
    items = load_items()
    
    search_query = st.text_input("Suchbegriff eingeben (z. B. Schlüssel, Handy, Brille):")
    
    filtered_items = [
        item for item in items 
        if search_query.lower() in item["kategorie"].lower() 
        or search_query.lower() in item["beschreibung"].lower()
        or search_query.lower() in item["ort"].lower()
    ]

    if filtered_items:
        for item in filtered_items:
            with st.expander(f"📦 {item['kategorie']} - Gefunden in {item['ort']}"):
                st.write(f"**Beschreibung:** {item['beschreibung']}")
                st.write(f"**Datum:** {item['datum']}")
                st.write(f"**Kontakt:** {item['kontakt']}")
    else:
        st.info("Keine passenden Gegenstände gefunden.")

# TAB 2: NEUES FUNDSTÜCK MELDEN
with tab2:
    st.header("Neues Fundstück registrieren")
    
    uploaded_file = st.file_uploader("Foto des Gegenstands hochladen", type=["jpg", "jpeg", "png"])
    
    detected_category = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Foto", width=250)
        
        # KI-Erkennung
        with st.spinner("KI analysiert das Bild..."):
            detected_category = predict_image(image)
        
        st.success(f"**KI-Erkennung:** {detected_category}")

    with st.form("new_item_form"):
        kategorie = st.text_input("Kategorie", value=detected_category)
        ort = st.text_input("Fundort (z. B. Park, Schule, Bus Line 5)")
        beschreibung = st.text_area("Zusätzliche Beschreibung (Farbe, Besondere Merkmale)")
        datum = st.date_input("Funddatum")
        kontakt = st.text_input("Kontakt-E-Mail oder Telefonnummer")
        
        submit = st.form_submit_button("Fundstück speichern")

        if submit:
            if kategorie and ort and kontakt:
                new_entry = {
                    "kategorie": kategorie,
                    "ort": ort,
                    "beschreibung": beschreibung,
                    "datum": str(datum),
                    "kontakt": kontakt
                }
                current_items = load_items()
                current_items.append(new_entry)
                save_items(current_items)
                st.success("Fundstück erfolgreich eingetragen!")
            else:
                st.error("Bitte fülle mindestens Kategorie, Ort und Kontakt aus.")
