import cv2
import numpy as np
import os
from PIL import Image
import streamlit as st
from io import StringIO

# Streamlit setup
st.title("Pizza Image Annotation Tool")
st.write("Upload a pizza image, annotate 'pizza-base' and 'pizza-pepperoni' with polygons, and download the YOLO format .txt file.")

# Upload image
uploaded_file = st.file_uploader("Choose a pizza image", type=["jpg", "jpeg", "png"])

# Annotation setup
class_ids = {'pizza-base': 0, 'pizza-pepperoni': 1}  # Class IDs
drawing = False
points = []
current_class = 0
image = None
window_name = "Annotate Pizza Image"
filename = None
temp_dir = "./temp"  # Temporary directory for image
output_dir = "./annotations"  # Directory for annotations
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
annotated_image = None  # To store the final annotated image

# Mouse callback function
def draw_polygon(event, x, y, flags, param):
    global drawing, points, image, filename, annotated_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Mark point
            points.append([x, y])
            if annotated_image is not None:
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)  # Update annotated image
            cv2.imshow(window_name, image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            drawing = False
            save_annotation()
            points = []

def save_annotation():
    global points, filename, current_class, annotated_image
    if points and filename and annotated_image is not None:
        h, w = image.shape[:2]
        # Normalize coordinates (0-1)
        normalized_points = [(x / w, y / h) for x, y in points]
        label_path = os.path.join(output_dir, os.path.basename(filename).replace(".jpg", ".txt"))
        with open(label_path, 'a') as f:
            poly_str = " ".join([f"{x} {y}" for x, y in normalized_points])
            f.write(f"{current_class} {poly_str}\n")
        # Draw the polygon on the annotated image
        points_array = np.array(points, np.int32)
        cv2.polylines(annotated_image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(annotated_image, [points_array], color=(0, 255, 0, 50))  # Semi-transparent fill
        st.write(f"Saved annotation for {os.path.basename(filename)} (class {list(class_ids.keys())[current_class]})")

# Download function
def download_txt(label_path):
    with open(label_path, 'r') as f:
        txt_content = f.read()
    return txt_content

# Main annotation loop
if uploaded_file is not None:
    # Display uploaded image
    input_img = Image.open(uploaded_file)
    st.image(input_img, caption="Uploaded Pizza Image", use_column_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    image = image_cv.copy()
    annotated_image = image_cv.copy()  # Initialize annotated image
    filename = os.path.join(temp_dir, uploaded_file.name)

    # Save the uploaded image temporarily
    cv2.imwrite(filename, image_cv)

    points = []
    drawing = True
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    while True:
        disp_image = image.copy()
        for pt in points:
            cv2.circle(disp_image, (pt[0], pt[1]), 2, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(disp_image, [np.array(points, np.int32)], False, (0, 255, 0), 1)
        
        cv2.imshow(window_name, disp_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save and close polygon
            if points:
                save_annotation()
                points = []
        elif key == ord('n'):  # Next class
            current_class = (current_class + 1) % len(class_ids)
            st.write(f"Switched to class: {list(class_ids.keys())[current_class]} (ID: {current_class})")
        elif key == ord('p'):  # Previous class
            current_class = (current_class - 1) % len(class_ids)
            st.write(f"Switched to class: {list(class_ids.keys())[current_class]} (ID: {current_class})")

    cv2.destroyWindow(window_name)
    
    # Display the annotated image
    if annotated_image is not None:
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        st.image(annotated_pil, caption="Annotated Pizza Image", use_column_width=True)
    
    # Provide download link for the .txt file
    label_path = os.path.join(output_dir, uploaded_file.name.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt"))
    if os.path.exists(label_path):
        txt_content = download_txt(label_path)
        st.download_button(
            label="Download Annotations (.txt)",
            data=txt_content,
            file_name=os.path.basename(label_path),
            mime="text/plain"
        )
    else:
        st.write("No annotations saved yet. Please annotate and save at least one polygon.")

    # Clean up temporary file
    if os.path.exists(filename):
        os.remove(filename)

    st.write("Annotation completed.")

# Optional: Display instructions
st.write("""
### Instructions:
- **Left-click**: Add a point to the polygon.
- **Right-click**: Finish and save the current polygon.
- **'s' key**: Save the polygon and start a new one.
- **'n' key**: Switch to the next class (pizza-base → pizza-pepperoni).
- **'p' key**: Switch to the previous class.
- **'q' key**: Quit and move to the next step.
- Annotate 'pizza-base' (class 0) for the entire dough and 'pizza-pepperoni' (class 1) for each slice.
""")