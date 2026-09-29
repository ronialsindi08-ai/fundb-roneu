import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# --- KONFIGURATION ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DATA_FILE = "fundbuero_data.json"

# Standard-Klassen als Fallback, falls labels.txt fehlen sollte
FALLBACK_LABELS = ["Helm", "Flasche", "sonstiges", "Turnbeutel"]


@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Fehler beim Laden des Modells ({MODEL_PATH}): {e}")
            return None
    return None


@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels_list = []
            for line in f.readlines():
                cleaned = line.strip()
                if cleaned:
                    # Falls Labels als "0 Helm" formatiert sind, Zahl vorn abschneiden
                    parts = cleaned.split(" ", 1)
                    if len(parts) > 1 and parts[0].isdigit():
                        labels_list.append(parts[1])
                    else:
                        labels_list.append(cleaned)
            if labels_list:
                return labels_list
    # Falls Datei nicht existiert oder leer ist, verwende die angegebenen Standard-Klassen
    return FALLBACK_LABELS


model = load_keras_model()
labels = load_labels()


# --- DATEN-HANDLING ---
def load_items():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_items(items):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# --- KI-VORHERSAGE ---
def predict_image(image):
    if model is None:
        return "Fehler: keras_model.h5 nicht gefunden", ""

    # Bild für Keras-Modell aufbereiten (Standard 224x224)
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Normalisierung (Teachable Machine / Keras Standard)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage
    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx]) * 100

    if class_idx < len(labels):
        detected_label = labels[class_idx]
        info_str = f"{detected_label} ({confidence:.1f}% Sicherheit)"
        return detected_label, info_str

    return "sonstiges", "Erkennung unsicher"


# --- STREAMLIT BENUTZEROBERFLÄCHE ---
st.set_page_config(page_title="KI-Fundbüro", page_icon="🔍", layout="centered")

st.title("🔍 KI-Fundbüro")
st.caption("Finde verlorene Gegenstände oder trage neue Fundstücke per KI-Fotoerkennung ein.")

tab1, tab2 = st.tabs(["🔎 Gegenstand suchen", "➕ Fundstück melden (mit KI)"])

# --- TAB 1: SUCHEN ---
with tab1:
    st.header("Fundstücke durchsuchen")
    items = load_items()

    search_query = st.text_input("Suchbegriff eingeben (z. B. Helm, Flasche, Turnbeutel):")

    if search_query:
        filtered_items = [
            item for item in items
            if search_query.lower() in item.get("kategorie", "").lower()
            or search_query.lower() in item.get("beschreibung", "").lower()
            or search_query.lower() in item.get("ort", "").lower()
        ]
    else:
        filtered_items = items

    if filtered_items:
        for item in reversed(filtered_items):  # Neueste zuerst anzeigen
            with st.expander(f"📦 {item.get('kategorie', 'Gegenstand')} — Fundort: {item.get('ort', 'Unbekannt')}"):
                st.write(f"**Beschreibung:** {item.get('beschreibung', '-')}")
                st.write(f"**Datum:** {item.get('datum', '-')}")
                st.write(f"**Kontakt:** {item.get('kontakt', '-')}")
    else:
        st.info("Keine passenden Fundstücke in der Datenbank vorhanden.")

# --- TAB 2: NEUES FUNDSTÜCK MELDEN ---
with tab2:
    st.header("Neues Fundstück registrieren")

    uploaded_file = st.file_uploader("Foto des Gegenstands hochladen", type=["jpg", "jpeg", "png"])

    detected_category = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Foto", width=250)

        with st.spinner("KI analysiert das Bild..."):
            detected_category, confidence_info = predict_image(image)

        if confidence_info:
            st.success(f"**KI-Ergebnis:** {confidence_info}")

    with st.form("new_item_form"):
        kategorie = st.text_input("Kategorie (von KI vorgeschlagen oder anpassen)", value=detected_category)
        ort = st.text_input("Fundort (z. B. Sporthalle, Pausenhof, Bus Linie 3)")
        beschreibung = st.text_area("Zusätzliche Beschreibung (Farbe, Marke, Aufschrift)")
        datum = st.date_input("Funddatum")
        kontakt = st.text_input("Kontakt (E-Mail oder Telefonnummer)")

        submit = st.form_submit_button("Fundstück im System speichern")

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
                st.success("✅ Das Fundstück wurde erfolgreich eingetragen!")
            else:
                st.error("Bitte fülle mindestens Kategorie, Fundort und Kontakt aus.")
