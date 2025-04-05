# app/camera.py

import cv2
import torch
from app.dehaze_model import load_dehazing_model, dehaze_image
from app.detection_model import model as yolo_model


# Load models
dehaze_model = load_dehazing_model()
#yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.classes = [0, 15, 16, 17]  # 0: person, 15–17: cat, dog, horse

# Start camera
def start_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dehazing
        dehazed_frame = dehaze_image(frame, dehaze_model)

        # YOLO detection
        results = yolo_model(dehazed_frame)
        detections = results.pandas().xyxy[0]

        count = 0
        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            if conf > 0.5:
                color = (0, 255, 0) if label == 'person' else (255, 0, 0)
                cv2.rectangle(dehazed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(dehazed_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                count += 1

        # Display count
        cv2.putText(dehazed_frame, f"Total Detected: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("Dehazed Human & Animal Detection", dehazed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
