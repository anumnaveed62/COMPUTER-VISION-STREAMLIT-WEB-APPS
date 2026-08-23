from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load a pretrained YOLOv8 model (nano version)
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for detection only (not 'yolov8n-seg.pt')

# Image source
source = r"C:\Users\Anam\Desktop\my--codes--working--portfolio\000000039769.jpg"

# Run prediction
results = model.predict(source=source, save=False)

# Loop through results
for r in results:
    frame = cv2.imread(source)

    # Each detected object
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        # Random color for each instance
        color = [random.randint(0, 255) for _ in range(3)]

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with class + confidence
        label = f"{model.names[int(cls_id)]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, lineType=cv2.LINE_AA)

    # Show result
    cv2.imshow("YOLOv8 Detection (Bounding Boxes Only)", frame)

    # Save result
    cv2.imwrite(r"C:\Users\Anam\Downloads\detected_boxes_only.jpg", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
