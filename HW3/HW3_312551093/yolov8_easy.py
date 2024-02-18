import cv2
from ultralytics import YOLO
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


specified_colors = [
    (255, 0, 0),   
    (0, 255, 0),   
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (128, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (0, 64, 64),
    (64, 0, 64),
    (64, 64, 64),
    (192, 0, 0),
    (0, 192, 0),
    (0, 0, 192),
    (192, 192, 0),
    (0, 192, 192),
    (192, 0, 192),
    (192, 192, 192),
    (32, 0, 0),
    (0, 32, 0),
    (0, 0, 32),
    (32, 32, 0),
    (0, 32, 32),
    (32, 0, 32),
    (32, 32, 32),
    (160, 0, 0),
    (0, 160, 0),
    (0, 0, 160),
    (160, 160, 0),
    (0, 160, 160),
    (160, 0, 160),
    (160, 160, 160),
    (96, 0, 0),
    (0, 96, 0),
    (0, 0, 96),
    (96, 96, 0),
    (0, 96, 96),
    (96, 0, 96),
    (96, 96, 96),
    (224, 0, 0),
    (0, 224, 0),
    (0, 0, 224),
    (224, 224, 0),
    (0, 224, 224),
    (224, 0, 224),
    (224, 224, 224),
    (16, 0, 0),
    (0, 16, 0),
    (0, 0, 16),
    (16, 16, 0),
    (0, 16, 16),
    (16, 0, 16),
    (16, 16, 16),
   
]
colors = set(specified_colors)
colors = list(colors)



# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# Open the video file
video_path = "easy_9.mp4"
output_path = "output_easy.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

while cap.isOpened():

    success, frame = cap.read()

    if success:
      
        results = model.track(frame, classes=0, conf=0.8, iou=0.5, persist=True,  tracker="bytetrack_easy.yaml")
        if results and results[0].boxes and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print("count: ", max(track_ids))
            annotated_frame = results[0].plot(colors = colors)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            out.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

    
        
        annotated_frame = results[0].plot(colors = colors)
      
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
      
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()