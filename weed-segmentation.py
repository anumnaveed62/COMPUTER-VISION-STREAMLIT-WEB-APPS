import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="NDVI Analyzer", layout="wide")
st.title("🌱 Multispectral Drone Imaging – NDVI Analyzer")
st.markdown("Upload an **RGB image** + a **NIR image** (grayscale or false-color). The app will automatically extract the correct bands.")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def compute_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.clip(ndvi, -1, 1)

def classify_health(ndvi):
    healthy = np.sum(ndvi > 0.6)
    stressed = np.sum((ndvi > 0.3) & (ndvi <= 0.6))
    unhealthy = np.sum(ndvi <= 0.3)
    total = ndvi.size
    return (healthy / total * 100,
            stressed / total * 100,
            unhealthy / total * 100)

def to_grayscale(img):
    """Convert any image to single-band grayscale"""
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        # If it looks like false-color NIR (strong red channel), use red channel
        # Otherwise just take the first channel or convert properly
        if img.shape[2] >= 3:
            # Heuristic: if red channel is much brighter on average → likely false-color NIR
            if np.mean(img[:, :, 0]) > np.mean(img[:, :, 1]) + 20:
                return img[:, :, 0]  # Use red channel as NIR
            else:
                return img[:, :, 0]  # Default to first channel
        else:
            return img[:, :, 0]
    else:
        raise ValueError("Unsupported image format")

# ------------------------------------------------------------
# File Uploaders
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    rgb_file = st.file_uploader("1. Upload RGB Image", type=["jpg", "jpeg", "png", "tif", "tiff"], key="rgb")

with col2:
    nir_file = st.file_uploader("2. Upload NIR Image (Grayscale or False-color)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="nir")

# ------------------------------------------------------------
# Main Processing
# ------------------------------------------------------------
if rgb_file and nir_file:
    # Load images
    rgb_img = np.array(Image.open(rgb_file).convert("RGB"))
    nir_img = np.array(Image.open(nir_file))

    # Display originals
    st.subheader("Uploaded Images")
    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb_img, caption="RGB Image", use_container_width=True)
    with c2:
        st.image(nir_img, caption="NIR Image (as uploaded)", use_container_width=True)

    # ----- Extract Red band from RGB -----
    red = rgb_img[:, :, 0]

    # ----- Extract / convert NIR band -----
    try:
        nir = to_grayscale(nir_img)
    except Exception as e:
        st.error(f"Could not process NIR image: {e}")
        st.stop()

    # Resize NIR to match RGB size if needed
    if nir.shape != red.shape:
        nir = np.array(Image.fromarray(nir).resize((red.shape[1], red.shape[0]), Image.BILINEAR))
        st.info(f"NIR image was resized from {nir_img.shape[:2]} → {red.shape} to match RGB.")

    # Show the extracted bands
    st.subheader("Extracted Bands")
    b1, b2 = st.columns(2)
    with b1:
        st.image(red, caption="Red Band (from RGB)", use_container_width=True, clamp=True)
    with b2:
        st.image(nir, caption="NIR Band (processed)", use_container_width=True, clamp=True)

    # ----- Compute NDVI -----
    ndvi = compute_ndvi(nir, red)

    # Display NDVI
    st.subheader("NDVI Map")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    ax.set_title("Normalized Difference Vegetation Index (NDVI)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="NDVI")
    st.pyplot(fig)

    # Health classification
    h, s, u = classify_health(ndvi)

    st.subheader("Vegetation Health Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("🌿 Healthy", f"{h:.1f}%")
    m2.metric("🌾 Stressed", f"{s:.1f}%")
    m3.metric("🥀 Unhealthy / Bare", f"{u:.1f}%")

    # Extra info
    with st.expander("Technical Details"):
        st.write(f"RGB shape: `{rgb_img.shape}`")
        st.write(f"Original NIR shape: `{nir_img.shape}`")
        st.write(f"Processed NIR shape: `{nir.shape}`")
        st.write(f"NDVI range: `{ndvi.min():.3f}` to `{ndvi.max():.3f}`")

else:
    st.info("Please upload both an RGB image and a NIR image to begin.")