from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="F:/Industrial-Defect-Detection/dataset/NEU-YOLO/data.yaml",
    epochs=10,
    imgsz=416,
    batch=2,
    workers=0,
    device="cpu",
    project="runs",
    name="defect_detection"
)

print("Training Completed!")