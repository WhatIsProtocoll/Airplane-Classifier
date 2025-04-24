import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os

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
        repo_id="SanderConn/Airplane_classifier7",  
        filename="model_21_04_yolo_crop_79_acc.keras",
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
    "A320", "A330", "A340", "A350", "A380", "ATR-72",
    "Boeing_737", "Boeing_737_MAX", "Boeing_747", "Boeing_757",
    "Boeing_767", "Boeing_777", "Boeing_787", "CRJ-700",
    "Dash_8", "Embraer_E-Jet"
]

# -----------------------------
# Demo image selection
# -----------------------------
st.subheader("📸 Try a Demo Image")
demo_folder = "data/demo_images"
demo_images = [f for f in os.listdir(demo_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
selected_demo = st.selectbox("Choose a demo image:", [""] + demo_images)

if selected_demo:
    uploaded_file = open(os.path.join(demo_folder, selected_demo), "rb")

# -----------------------------
# Image upload
# -----------------------------
st.markdown("### Or Upload Your Own Image")
uploaded_file = uploaded_file or st.file_uploader("Upload an airport image", type=["jpg", "jpeg", "png"])

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
            pred = clf_model.predict(img_input)[0]
            top_k = tf.nn.top_k(pred, k=3)
            top_k_indices = top_k.indices.numpy()
            top_k_values = top_k.values.numpy()

            top_k_preds = [f"{class_labels[idx]} ({conf:.2f})" for idx, conf in zip(top_k_indices, top_k_values)]
            predictions.append(top_k_preds)
            crops.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    # -----------------------------
    # Display results
    # -----------------------------
    if crops:
        st.subheader("✂️ Cropped Airplanes and Predicted Families")
        cols = st.columns(len(crops))
        for idx, col in enumerate(cols):
            col.image(crops[idx], use_container_width=True)
            col.markdown("**Top 3 predictions:**")
            for line in predictions[idx]:
                col.markdown(f"- {line}")
    else:
        st.warning("No airplanes detected.")
