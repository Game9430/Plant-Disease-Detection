import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import uuid
from deep_translator import GoogleTranslator

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ----------------------------- CUSTOM CSS -----------------------------
def local_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

local_css("""
/* Main container */
.main {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 20px;
    border-radius: 15px;
}

/* Header */
h1 {
    text-align: center;
    font-size: 45px !important;
    font-weight: 900 !important;
    color: #2e7d32 !important;
    margin-bottom: 10px;
}

/* Markdown */
.stMarkdown, p {
    font-size: 18px !important;
}

/* File upload box */
.css-1y4p8pa {
    border: 2px dashed #66bb6a !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Buttons */
.stButton>button {
    background: #43a047 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: 0.3s ease-in-out;
}
.stButton>button:hover {
    background: #2e7d32 !important;
    transform: scale(1.05);
}

/* ---------------- Dropdown Fix: Ensures Text Is Visible ---------------- */
/* FORCE SELECTBOX BACKGROUND STAY WHITE IN ANY THEME */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;   /* white background */
    color: #000000 !important;              /* black text */
    border: 2px solid #81c784 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] * {
    background-color: #ffffff !important;   /* Fix inside elements */
    color: #000000 !important;              /* Make all text black */
}

.stSelectbox [data-baseweb="popover"] {
    background-color: #ffffff !important;   /* dropdown list background */
}

.stSelectbox [data-baseweb="option"] {
    color: #000000 !important;              /* option text color */
    background-color: #ffffff !important;   /* option background */
}

.stSelectbox svg {
    fill: #2e7d32 !important;               /* green dropdown arrow */
}


/* Image border */
.stImage img {
    border-radius: 12px !important;
    border: 3px solid #66bb6a !important;
}

/* Alert Box */
.stAlert {
    border-radius: 12px !important;
    font-size: 18px !important;
}

/* Page centering */
.block-container {
    max-width: 700px;
    margin: auto;
    padding-top: 2rem;
}

/* Footer */
footer {
    text-align: center !important;
    color: gray !important;
    margin-top: 30px !important;
}
""")

# ----------------------------- DIRECTORIES -----------------------------
MODEL_DIR = r"/workspaces/Plant-Disease-Detection/models/"
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)

PLANTS = [
    "apple", "tomato", "potato", "orange", "chili", "grape", "tea",
    "peach", "coffee", "corn", "cucumber", "jamun", "lemon",
    "mango", "pepper", "rice", "soybean", "sugarcane", "wheat"
]

# ----------------------------- CACHING -----------------------------
@st.cache_resource
def load_class_and_prevention_maps():
    inv_maps = {}
    prevention_maps = {}

    for plant in PLANTS:
        class_map_path = os.path.join(MODEL_DIR, f"{plant}_class_map.json")
        prevention_path = os.path.join(MODEL_DIR, f"{plant}_prevention.json")

        # Load class map
        if os.path.exists(class_map_path):
            with open(class_map_path, "r") as f:
                class_map = json.load(f)
            inv_maps[plant] = {v: k for k, v in class_map.items()}
        else:
            inv_maps[plant] = {}
            st.warning(f"⚠️ Class map missing for {plant}")

        # Load prevention file
        if os.path.exists(prevention_path):
            with open(prevention_path, "r") as f:
                prevention_maps[plant] = json.load(f)
        else:
            prevention_maps[plant] = {}
            st.warning(f"⚠️ Prevention file missing for {plant}")

    return inv_maps, prevention_maps


# ----------------------------- MODEL LOADER -----------------------------
@st.cache_resource
def load_plant_model(plant_name: str):
    model_path = os.path.join(MODEL_DIR, f"{plant_name}.h5")
    if os.path.exists(model_path):
        return keras_load_model(model_path)
    else:
        st.error(f"Model not found: {model_path}")
        return None


inv_maps, prevention_maps = load_class_and_prevention_maps()

# ----------------------------- TRANSLATION -----------------------------
def translate_text(text: str, lang_code: str = "en"):
    if lang_code == "en" or not text:
        return text
    try:
        return GoogleTranslator(source='en', target=lang_code).translate(text)
    except:
        return text

# ----------------------------- UI -----------------------------
st.title("🌱 Plant Disease Detection")
st.markdown("Upload a leaf image to detect disease and get prevention tips.")

# Language selection
lang_options = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Bengali": "bn", "Tamil": "ta", "Telugu": "te"
}

selected_lang_name = st.selectbox("Select Language / भाषा चुनें", options=list(lang_options.keys()))
lang_code = lang_options[selected_lang_name]

# Plant selection
plant = st.selectbox(
    "Select Plant / पौधा चुनें",
    options=PLANTS,
    format_func=lambda x: x.capitalize()
)

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and plant:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("Detect Disease"):
        with st.spinner("Analyzing image..."):
            model = load_plant_model(plant)

            if model is None:
                st.error("Model could not be loaded.")
                st.stop()

            img_resized = img.resize(IMAGE_SIZE)
            x = image.img_to_array(img_resized) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x)[0]
            idx = np.argmax(pred)
            confidence = float(pred[idx])
            predicted_class = inv_maps.get(plant, {}).get(idx, "Unknown Disease")

            prevention = prevention_maps.get(plant, {}).get(
                predicted_class, "No prevention advice available."
            )

            disease_translated = translate_text(predicted_class.replace("_", " "), lang_code)
            prevention_translated = translate_text(prevention, lang_code)

            # Save uploaded image
            filename = f"{uuid.uuid4()}.jpg"
            img.save(os.path.join(UPLOAD_DIR, filename))

        st.success(f"**Detected: {disease_translated}**")
        st.write(f"**Confidence:** {confidence:.2%}")

        if confidence < 0.5:
            st.warning("Low confidence prediction — image quality may affect accuracy.")

        st.subheader("Prevention & Cure Tips")
        st.info(prevention_translated)

        if st.checkbox("Show detailed probabilities"):
            probs = {inv_maps[plant].get(i, f"Class {i}"): float(p) for i, p in enumerate(pred)}
            st.json(probs)

else:
    st.info("Please select a plant and upload a clear leaf image.")

# ----------------------------- FOOTER -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | AI-based Plant Disease Detection")
