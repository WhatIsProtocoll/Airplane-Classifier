import streamlit as st
import cv2
#import os
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(layout="wide")
st.title("✈️ Airport Airplane Detector & Classifier")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img_cv2_bgr, target_size=(224, 224)):
    img = cv2.resize(img_cv2_bgr, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    # Load YOLOv8
    yolo = YOLO('yolov8n.pt')

    # Download model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="SanderConn/dense82acc",  # ⚠️ deinen Hugging Face Namen + Repo anpassen
        filename="model_19_04_densenet_82_acc.keras",
        repo_type="model",
        force_download=True
    )
    clf = load_model(model_path)
    return yolo, clf

yolo_model, clf_model = load_models()

# -----------------------------
# Class labels (update as needed)
# -----------------------------
class_labels = [
    'A330', 'Boeing_747', 'CRJ-700', 'ATR-72', 'Boeing_777', 'Boeing_757', 
    'Boeing_767', 'A340', 'Boeing_737', 'Embraer_E-Jet', 'A320', 
    'A380', 'Dash_8', 'A350', 'Boeing_787', 'Boeing_737_MAX'
]

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an airport image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

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
            predictions.append((pred_idx, pred_label, pred_conf))


    # -----------------------------
    # Display results
    # -----------------------------
    if crops:
        st.subheader("✂️ Cropped Airplaness and Predicted Families")
        cols = st.columns(len(crops))
        for idx, col in enumerate(cols):
            class_idx, class_name, confidence = predictions[idx]
col.image(
    crops[idx],
    caption=f"Class {class_idx}: {class_name} ({confidence:.2f})",
    use_container_width=True
)


    else:
        st.warning("No airplanes detected.")
