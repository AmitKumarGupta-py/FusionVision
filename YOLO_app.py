import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit as st

# Load YOLOv8 Model
yolo_model = YOLO('yolov8n.pt')

# Object Detection Function
def detect_objects(image):
    results = yolo_model(image)
    for result in results:
        annotated_frame = result.plot()  # Draw bounding boxes
    return annotated_frame

# Streamlit UI
st.title("🔍 YOLOv8 Object Detection")

# YOLO Object Detection Section
st.header("🖼 Upload an Image for Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    detected_image = detect_objects(image)
    st.image(detected_image, caption="Detected Objects", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("🚀 **Built with YOLOv8 and Streamlit**")
