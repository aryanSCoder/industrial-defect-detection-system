import requests
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "dataset/NEU-DET/IMAGES/crazing_1.jpg"
PLC_URL = "http://127.0.0.1:5001/plc/detection"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 416

CLASS_NAMES = {
    0: "crazing",
    1: "patches",
    2: "inclusion",
    3: "pitted_surface",
    4: "scratches",
    5: "rolled-in_scale"
}


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")


# ==============================
# RUN DETECTION
# ==============================

print("\nRunning defect detection...")

results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE_THRESHOLD,
    imgsz=IMAGE_SIZE,
    verbose=False
)

result = results[0]

# ==============================
# PROCESS DETECTIONS
# ==============================

detections = result.boxes

if detections is not None and len(detections) > 0:

    # Select highest-confidence detection
    best_index = detections.conf.argmax().item()

    confidence = float(detections.conf[best_index].item())
    class_id = int(detections.cls[best_index].item())

    # Bounding box
    box = detections.xyxy[best_index].cpu().numpy()

    x1, y1, x2, y2 = box

    # Center coordinates
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    defect_type = CLASS_NAMES.get(
        class_id,
        result.names.get(class_id, "unknown")
    )

    print("\n========== DETECTION ==========")
    print(f"Defect       : {defect_type}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"X Coordinate : {x_center}")
    print(f"Y Coordinate : {y_center}")
    print("===============================")

    plc_data = {
        "defect_detected": True,
        "defect_type": defect_type,
        "confidence": round(confidence * 100, 2),
        "x_coordinate": x_center,
        "y_coordinate": y_center
    }

else:

    print("\nNo defect detected.")

    plc_data = {
        "defect_detected": False,
        "defect_type": None,
        "confidence": 0,
        "x_coordinate": None,
        "y_coordinate": None
    }


# ==============================
# SEND RESULT TO PLC
# ==============================

print("\nSending result to PLC...")

try:

    response = requests.post(
        PLC_URL,
        json=plc_data,
        timeout=5
    )

    print(f"PLC Status Code : {response.status_code}")

    try:
        print("PLC Response:")
        print(response.json())
    except Exception:
        print(response.text)

except requests.exceptions.ConnectionError:

    print("\nERROR: PLC simulator is not running.")
    print("Start plc_simulator.py first.")

except requests.exceptions.Timeout:

    print("\nERROR: PLC request timed out.")

except Exception as e:

    print(f"\nPLC ERROR: {e}")


print("\nYOLO → PLC integration completed.")