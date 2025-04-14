import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("✈️ Airport Airplane Detector & Classifier")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img_cv2_bgr, target_size=(299, 299)):
    img = cv2.resize(img_cv2_bgr, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    clf = load_model('/content/drive/MyDrive/Dat255/model_06_04.keras')
    return yolo, clf

yolo_model, clf_model = load_models()

# -----------------------------
# Class labels (update as needed)
# -----------------------------
class_labels = [
    "A320", "A330", "A380", "A340", "ATR-72", "Boeing 737", "Boeing 747",
    "Boeing 757", "Boeing 767", "Boeing 777", "CRJ-700", "Dash 8",
    "Embraer E-Jet", "Airbus A350", "Boeing 787"
]

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an airport image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Detecting airplanes...")

    # -----------------------------
    # YOLO inference
    # -----------------------------
    results = yolo_model.predict(image_bgr, conf=0.1, iou=0.2)
    result = results[0]

    # Find class ID for "airplane"
    airplane_id = [k for k, v in yolo_model.names.items() if v == 'airplane'][0]

    crops = []
    predictions = []

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == airplane_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = image_bgr.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = image_bgr[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            # Preprocess and predict
            img_input = preprocess_image(cropped)
            pred = clf_model.predict(img_input)
            pred_idx = np.argmax(pred)
            pred_label = class_labels[pred_idx]
            pred_conf = float(np.max(pred))

            crops.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))  # For display
            predictions.append((pred_label, pred_conf))

    # -----------------------------
    # Display results
    # -----------------------------
    if crops:
        st.subheader("✂️ Cropped Airplanes and Predicted Families")
        cols = st.columns(len(crops))
        for idx, col in enumerate(cols):
            col.image(crops[idx], caption=f"{predictions[idx][0]} ({predictions[idx][1]:.2f})", use_column_width=True)
    else:
        st.warning("No airplanes detected.")