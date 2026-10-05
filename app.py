import os
import json
from PIL import Image
import streamlit as st
from transformers import pipeline

# --- DATEN-HANDLING ---
DATA_FILE = "fundbuero_data.json"

# Deine gewünschten Klassen für das Fundbüro
CATEGORIES = ["Helm", "Flasche", "Turnbeutel", "sonstiges"]


@st.cache_resource
def load_hf_model():
    try:
        # Lädt ein Zero-Shot Bildklassifizierungsmodell von Hugging Face
        classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )
        return classifier
    except Exception as e:
        st.error(f"Fehler beim Laden des Hugging Face Modells: {e}")
        return None


classifier = load_hf_model()


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


# --- KI-VORHERSAGE MIT HUGGING FACE ---
def predict_image(image):
    if classifier is None:
        return "sonstiges", "Hugging Face Modell nicht bereit"

    # Bild zu RGB konvertieren
    img = image.convert("RGB")

    # KI Vorhersage über Hugging Face Pipeline
    results = classifier(img, candidate_labels=CATEGORIES)

    # Bestes Ergebnis extrahieren
    top_result = results[0]
    detected_label = top_result["label"]
    confidence = top_result["score"] * 100

    info_str = f"{detected_label} ({confidence:.1f}% Sicherheit)"
    return detected_label, info_str


# --- STREAMLIT BENUTZEROBERFLÄCHE ---
st.set_page_config(page_title="KI-Fundbüro", page_icon="🔍", layout="centered")

st.title("🔍 KI-Fundbüro (mit Hugging Face KI)")
st.caption("Finde verlorene Gegenstände oder trage neue Fundstücke per Fotoerkennung ein.")

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
        for item in reversed(filtered_items):  # Neueste zuerst
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

        with st.spinner("Hugging Face KI analysiert das Bild..."):
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
