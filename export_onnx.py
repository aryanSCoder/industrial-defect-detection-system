from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Converting model to ONNX...")

onnx_path = model.export(
    format="onnx",
    imgsz=416,
    dynamic=True,
    simplify=True
)

print("\nONNX conversion completed successfully!")
print(f"ONNX model saved at: {onnx_path}")