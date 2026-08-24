import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.title("🌱 Multispectral Drone Imaging for Precision Agriculture")

# Option to choose upload type
upload_option = st.radio(
    "Choose upload method:",
    ("Upload Multispectral (RGB+NIR)", "Upload RGB + Separate NIR")
)

# Function for NDVI
def compute_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

# Function for classification
def classify_health(ndvi):
    healthy = np.sum(ndvi > 0.6)
    stressed = np.sum((ndvi > 0.3) & (ndvi <= 0.6))
    unhealthy = np.sum(ndvi <= 0.3)
    total = ndvi.size
    return healthy/total*100, stressed/total*100, unhealthy/total*100

# --- CASE 1: Upload a single multispectral image ---
if upload_option == "Upload Multispectral (RGB+NIR)":
    uploaded_file = st.file_uploader("Upload a Multispectral Image (RGB+NIR)", type=["jpg","png","tif"])

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Multispectral Image", use_column_width=True)

        if image.shape[2] >= 4:
            red = image[:,:,0]
            nir = image[:,:,3]

            ndvi = compute_ndvi(nir, red)

            st.subheader("NDVI Map")
            fig, ax = plt.subplots()
            cax = ax.imshow(ndvi, cmap="RdYlGn")
            fig.colorbar(cax)
            st.pyplot(fig)

            h, s, u = classify_health(ndvi)
            st.write(f"🌿 Healthy Crops: {h:.2f}%")
            st.write(f"🌾 Stressed Crops: {s:.2f}%")
            st.write(f"🥀 Unhealthy Crops: {u:.2f}%")
        else:
            st.warning("Upload a multispectral image with at least 4 channels (RGB+NIR).")

# --- CASE 2: Upload RGB + Separate NIR image ---
else:
    rgb_file = st.file_uploader("Upload RGB Image", type=["jpg","png","tif"], key="rgb")
    nir_file = st.file_uploader("Upload Separate NIR Image", type=["jpg","png","tif"], key="nir")

    if rgb_file and nir_file:
        rgb_image = np.array(Image.open(rgb_file))
        nir_image = np.array(Image.open(nir_file))

        st.image(rgb_image, caption="Uploaded RGB Image", use_column_width=True)
        st.image(nir_image, caption="Uploaded NIR Image", use_column_width=True)

        # Use red channel from RGB + separate NIR
        if len(rgb_image.shape) == 3:
            red = rgb_image[:,:,0]
        else:
            st.error("RGB image is not valid (must have 3 channels).")

        if len(nir_image.shape) == 2:
            nir = nir_image
        else:
            st.error("NIR image must be single-band (grayscale).")

        # Compute NDVI if valid
        if len(rgb_image.shape) == 3 and len(nir_image.shape) == 2:
            ndvi = compute_ndvi(nir, red)

            st.subheader("NDVI Map")
            fig, ax = plt.subplots()
            cax = ax.imshow(ndvi, cmap="RdYlGn")
            fig.colorbar(cax)
            st.pyplot(fig)

            h, s, u = classify_health(ndvi)
            st.write(f"🌿 Healthy Crops: {h:.2f}%")
            st.write(f"🌾 Stressed Crops: {s:.2f}%")
            st.write(f"🥀 Unhealthy Crops: {u:.2f}%")
