import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import pickle
import tempfile
import torch
from ultralytics import YOLO
import numpy as np
import io
import pandas as pd
from datetime import datetime

# --------------------- Page Setup ---------------------
st.set_page_config(page_title="LiDAR Detection App", layout="wide")
st.title("🚗 LiDAR Object Detection using YOLO")
st.markdown("Upload a LiDAR image to get object detections using a pre-trained YOLO model.")

# --------------------- Constants ---------------------
MODEL_PATH = "lidar_complete_model.pkl"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}
colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

# --------------------- Helper Functions ---------------------

@st.cache_resource
def load_model(pickle_path):
    with open(pickle_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict(model_data, image_path, conf_thresh=0.5):
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        torch.save(model_data['complete_model_state'], tmp.name)
        model = YOLO(tmp.name)

    results = model.predict(source=image_path, conf=conf_thresh, save=False, show=False)
    detections = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            class_name = model_data['class_names'][class_id]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })
    os.remove(tmp.name)
    return detections

def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for det in detections:
        box = det["bbox"]
        color = colors[det["class_id"] % len(colors)]
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        draw.rectangle(box, outline=color, width=3)

        # Draw label background
        text_size = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle([box[0], box[1] - 20, box[0] + (text_size[2] - text_size[0]) + 5, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - 20), label, fill="white", font=font)

    return image

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_table = st.checkbox("Show Detection Table", True)
    st.markdown("---")
    st.caption("Model: `lidar_complete_model.pkl`")

# --------------------- Upload Section ---------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=ALLOWED_EXTENSIONS)

if uploaded_file and allowed_file(uploaded_file.name):
    with st.spinner("Loading model and running predictions..."):
        # Save uploaded file to a temporary location
        img_bytes = uploaded_file.read()
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Load model and run prediction
        model_data = load_model(MODEL_PATH)
        detections = predict(model_data, temp_path, conf_thresh=confidence)

        # Display uploaded image and result
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result_image = draw_detections(image.copy(), detections)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📥 Uploaded Image")
        with col2:
            st.image(result_image, caption="📌 Detected Objects")

        # Show detection table
        if show_table and detections:
            st.markdown("### 📋 Detection Results")
            st.dataframe(pd.DataFrame(detections))
        elif not detections:
            st.warning("⚠️ No objects detected in this image.")
else:
    st.info("Please upload a valid LiDAR image file.")
