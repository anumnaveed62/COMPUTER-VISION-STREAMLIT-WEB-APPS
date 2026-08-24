# Import required libraries
import streamlit as st        # Streamlit for web app interface
import numpy as np            # NumPy for numerical and array operations
import cv2                    # OpenCV (optional here, could be used for image processing)
from PIL import Image         # Pillow for opening image files
import matplotlib.pyplot as plt  # Matplotlib for plotting NDVI map

# Title of the web app
st.title("🌱 Multispectral Drone Imaging for Precision Agriculture")

# Radio button to select upload type (either one combined image or separate RGB and NIR)
upload_option = st.radio(
    "Choose upload method:", 
    ("Upload Multispectral (RGB+NIR)", "Upload RGB + Separate NIR")
)

# ------------------------------------------------------------
# Function to compute NDVI (Normalized Difference Vegetation Index)
# ------------------------------------------------------------
def compute_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)   # Convert NIR band to float for accurate division
    red = red_band.astype(float)   # Convert Red band to float
    ndvi = (nir - red) / (nir + red + 1e-6)  # NDVI formula; add small value to avoid division by zero
    return ndvi                    # Return NDVI array

# ------------------------------------------------------------
# Function to classify vegetation health based on NDVI thresholds
# ------------------------------------------------------------
def classify_health(ndvi):
    healthy = np.sum(ndvi > 0.6)                           # Count pixels with NDVI > 0.6 (healthy vegetation)
    stressed = np.sum((ndvi > 0.3) & (ndvi <= 0.6))        # NDVI between 0.3 and 0.6 = stressed vegetation
    unhealthy = np.sum(ndvi <= 0.3)                        # NDVI ≤ 0.3 = unhealthy or bare soil
    total = ndvi.size                                      # Total number of pixels
    return healthy/total*100, stressed/total*100, unhealthy/total*100  # Return percentage for each category

# ------------------------------------------------------------
# CASE 1: User uploads a single multispectral image (RGB + NIR in one file)
# ------------------------------------------------------------
if upload_option == "Upload Multispectral (RGB+NIR)":
    # Upload the multispectral image (must contain 4 channels)
    uploaded_file = st.file_uploader("Upload a Multispectral Image (RGB+NIR)", type=["jpg","png","tif"])

    if uploaded_file:   # Check if the user uploaded a file
        image = np.array(Image.open(uploaded_file))  # Read image as NumPy array
        st.image(image, caption="Uploaded Multispectral Image", use_column_width=True)  # Display uploaded image

        if image.shape[2] >= 4:    # Check if image has at least 4 bands (R, G, B, NIR)
            red = image[:,:,0]     # Extract Red channel (band 1)
            nir = image[:,:,3]     # Extract NIR channel (band 4)

            ndvi = compute_ndvi(nir, red)   # Compute NDVI using NIR and Red bands

            # Display NDVI map using matplotlib
            st.subheader("NDVI Map")
            fig, ax = plt.subplots()             # Create a figure and axis
            cax = ax.imshow(ndvi, cmap="RdYlGn") # Show NDVI with Red-Yellow-Green color scale
            fig.colorbar(cax)                    # Add a colorbar legend
            st.pyplot(fig)                       # Render the plot in Streamlit

            # Classify crop health
            h, s, u = classify_health(ndvi)
            st.write(f"🌿 Healthy Crops: {h:.2f}%")     # Display healthy vegetation percentage
            st.write(f"🌾 Stressed Crops: {s:.2f}%")    # Display stressed vegetation percentage
            st.write(f"🥀 Unhealthy Crops: {u:.2f}%")   # Display unhealthy vegetation percentage

        else:
            # Warning if image doesn’t have enough channels
            st.warning("Upload a multispectral image with at least 4 channels (RGB+NIR).")

# ------------------------------------------------------------
# CASE 2: User uploads RGB image and NIR image separately
# ------------------------------------------------------------
else:
    # File uploaders for RGB and NIR images
    rgb_file = st.file_uploader("Upload RGB Image", type=["jpg","png","tif"], key="rgb")
    nir_file = st.file_uploader("Upload Separate NIR Image", type=["jpg","png","tif"], key="nir")

    # Proceed only if both files are uploaded
    if rgb_file and nir_file:
        rgb_image = np.array(Image.open(rgb_file))  # Read RGB image
        nir_image = np.array(Image.open(nir_file))  # Read NIR image

        # Display uploaded images
        st.image(rgb_image, caption="Uploaded RGB Image", use_column_width=True)
        st.image(nir_image, caption="Uploaded NIR Image", use_column_width=True)

        # Validate and extract bands
        if len(rgb_image.shape) == 3:    # RGB image must have 3 channels
            red = rgb_image[:,:,0]       # Extract Red band from RGB
        else:
            st.error("RGB image is not valid (must have 3 channels).")

        if len(nir_image.shape) == 2:    # NIR image must be single-band (grayscale)
            nir = nir_image
        else:
            st.error("NIR image must be single-band (grayscale).")

        # Compute NDVI only if both images are valid
        if len(rgb_image.shape) == 3 and len(nir_image.shape) == 2:
            ndvi = compute_ndvi(nir, red)   # Compute NDVI

            # Display NDVI map
            st.subheader("NDVI Map")
            fig, ax = plt.subplots()             # Create figure
            cax = ax.imshow(ndvi, cmap="RdYlGn") # NDVI color visualization
            fig.colorbar(cax)                    # Add color legend
            st.pyplot(fig)                       # Show figure in Streamlit

            # Classify vegetation health
            h, s, u = classify_health(ndvi)
            st.write(f"🌿 Healthy Crops: {h:.2f}%")
            st.write(f"🌾 Stressed Crops: {s:.2f}%")
            st.write(f"🥀 Unhealthy Crops: {u:.2f}%")
