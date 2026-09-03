from ultralytics import YOLO
import cv2
import os

MODEL_PATH = r"runs\detect\runs\defect_detection-4\weights\best.pt"
IMAGE_PATH = r"dataset\NEU-DET\IMAGES\crazing_1.jpg"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True
)

for result in results:
    print("\nDetection completed successfully!")

    if result.boxes is not None and len(result.boxes) > 0:
        print(f"Number of defects detected: {len(result.boxes)}")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"Defect: {class_name} | "
                f"Confidence: {confidence:.2%}"
            )
    else:
        print("No defects detected.")