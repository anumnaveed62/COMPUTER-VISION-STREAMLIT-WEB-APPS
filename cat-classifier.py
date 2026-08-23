# -------------------- Import Libraries -------------------- #
import streamlit as st           # Streamlit for web interface
from ultralytics import YOLO     # YOLOv8 model from Ultralytics
import cv2                       # OpenCV for image processing
import numpy as np               # NumPy for arrays
import random                    # Random for color generation
from PIL import Image            # PIL for handling uploaded images
import tempfile                  # Temporary directory for processing

# -------------------- Streamlit Page Setup -------------------- #
st.set_page_config(page_title="YOLOv8 Segmentation App", layout="wide")
st.title("🎯 YOLOv8 Object Detection & Segmentation")
st.write("Upload an image and visualize **object detection with colored segmentation masks** using YOLOv8!")

# -------------------- Sidebar -------------------- #
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# -------------------- Load YOLOv8 Model -------------------- #
@st.cache_resource
def load_model():
    # Load the lightweight YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")
    return model

model = load_model()

# -------------------- If Image Uploaded -------------------- #
if uploaded_file is not None:
    # Display uploaded image
    st.subheader("📸 Original Image")
    input_img = Image.open(uploaded_file)
    st.image(input_img, use_container_width=True)

    # Convert uploaded image (PIL → OpenCV BGR)
    image_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    # Save temporarily (YOLO expects file path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, image_cv)

    # Run YOLOv8 prediction
    st.subheader("🔍 Running YOLOv8 Segmentation...")
    results = model.predict(source=temp_file.name, save=False)

    # Process results
    for r in results:
        frame = image_cv.copy()  # Original image copy for drawing

        # Check if results have m


