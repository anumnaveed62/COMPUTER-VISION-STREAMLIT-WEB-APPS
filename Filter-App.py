# ---------------- IMPORT LIBRARIES ---------------- #
import streamlit as st       # Streamlit for creating the web app interface
import numpy as np           # NumPy for numerical operations and image arrays
import cv2                   # OpenCV for image processing filters
from PIL import Image        # PIL for handling image uploads and conversions


# ---------------- HELPER FUNCTION ---------------- #
# Function used in the pencil sketch filter for blending two images
def dodgeV2(x, y):
    # Divide the gray image by the inverted blurred image for sketch effect
    return cv2.divide(x, 255 - y, scale=256)


# ---------------- PENCIL SKETCH FILTER ---------------- #
def pencilsketch(inp_img):
    # Convert image from BGR to grayscale
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image (light becomes dark and vice versa)
    img_invert = cv2.bitwise_not(img_gray)
    
    # Apply Gaussian blur to smooth the inverted image
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    
    # Blend the grayscale image with the blurred inverted image
    final_img = dodgeV2(img_gray, img_smoothing)
    
    # Return the final pencil sketch image
    return final_img


# ---------------- CARTOON FILTER ---------------- #
def cartoonize(inp_img):
    # Convert image to grayscale
    gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise and smooth edges
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  # Edge detection parameters
        cv2.THRESH_BINARY, 9, 9
    )
    
    # Smooth color regions while keeping edges sharp using bilateral filter
    color = cv2.bilateralFilter(inp_img, 9, 300, 300)
    
    # Combine color image with edges to create a cartoon effect
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    # Return cartoon-style image
    return cartoon


# ---------------- SEPIA FILTER ---------------- #
def sepia_filter(inp_img):
    # Define a sepia tone transformation matrix (BGR channels)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    
    # Apply the sepia color transformation
    sepia = cv2.transform(inp_img, kernel)
    
    # Clip pixel values to stay within valid range (0–255)
    sepia = np.clip(sepia, 0, 255)
    
    # Convert back to unsigned 8-bit integer format
    return sepia.astype(np.uint8)


# ---------------- INVERT / NEGATIVE FILTER ---------------- #
def invert_filter(inp_img):
    # Invert all pixel values to create a negative image
    return cv2.bitwise_not(inp_img)


# ---------------- STREAMLIT WEB APP ---------------- #
# App title shown at the top
st.title("Photo Filter App 🎨")

# Description text under title
st.write("This app applies **multiple artistic filters** to your uploaded photo!")

# Sidebar for uploading an image (supports jpg, jpeg, png formats)
file_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# ---------------- MAIN APP LOGIC ---------------- #
if file_image is None:
    # If no image is uploaded, show a prompt message
    st.write("👉 Please upload an image file to continue.")
else:
    # Open the uploaded image with PIL
    input_img = Image.open(file_image)

    # Convert from PIL (RGB format) to OpenCV (BGR format)
    input_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    # Apply all the different filters
    sketch_img = pencilsketch(input_cv)     # Pencil sketch effect
    cartoon_img = cartoonize(input_cv)      # Cartoon-style effect
    sepia_img = sepia_filter(input_cv)      # Sepia (vintage tone)
    negative_img = invert_filter(input_cv)  # Negative / inverted image

    # Display the original uploaded image
    st.write("### Original Photo")
    st.image(input_img, use_container_width=True)

    # Display all filtered results in a 2-column layout
    st.write("### Filter Results")
    col1, col2 = st.columns(2)  # Create two columns for displaying images side by side

    with col1:
        # Show pencil sketch and sepia images
        st.image(sketch_img, caption="Pencil Sketch", use_container_width=True, clamp=True)
        st.image(sepia_img, caption="Sepia", use_container_width=True)
    
    with col2:
        # Show cartoon and negative images
        st.image(cartoon_img, caption="Cartoon", use_container_width=True)
        st.image(negative_img, caption="Negative", use_container_width=True)

# ---------------- FOOTER ---------------- #
# A closing note at the bottom of the app
st.write("This web app applies **sketch, cartoon, sepia, and negative filters** 🎭")
