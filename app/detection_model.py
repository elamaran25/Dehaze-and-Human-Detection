# app/detection_model.py

import torch
import cv2

# Load the YOLOv5 model from local folder
model = torch.hub.load('models_weights/model_weights/yolov5', 'yolov5s', source='local')  # you can change to yolov5m/yolov5x

# Only detect 'person', 'cat', 'dog', etc. (adjust as needed)
def detect_objects(image):
    # Convert image from BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model(img_rgb)

    # Get bounding boxes and labels
    detections = results.pandas().xyxy[0]

    # Count
    human_count = 0
    animal_count = 0

    for i in range(len(detections)):
        label = detections.iloc[i]['name']
        if label == 'person':
            human_count += 1
        elif label in ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep']:  # expand if needed
            animal_count += 1
        
        # Draw box
        x1, y1, x2, y2 = map(int, detections.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image, human_count, animal_count
