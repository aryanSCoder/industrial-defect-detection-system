from ultralytics import YOLO

MODEL_PATH = "models/best.onnx"
IMAGE_PATH = "dataset/NEU-DET/IMAGES/crazing_1.jpg"

print("Loading ONNX model...")

model = YOLO(MODEL_PATH)

print("Running prediction...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    imgsz=416,
    save=True,
    verbose=False
)

print("\nONNX prediction completed successfully!")

result = results[0]

total_defects = 0

if result.boxes is not None:
    total_defects = len(result.boxes)

print(f"Number of defects detected: {total_defects}")

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Defect: {model.names[class_id]} | "
            f"Confidence: {confidence * 100:.2f}%"
        )