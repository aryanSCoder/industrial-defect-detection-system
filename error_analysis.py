import os
import cv2
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\best.pt"

IMAGE_DIR = r"dataset\NEU-DET\IMAGES"
ANNOTATION_DIR = r"dataset\NEU-DET\ANNOTATIONS"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

OUTPUT_DIR = r"runs\error_analysis"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "error_analysis.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
print("=" * 60)

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "crazing",
    1: "patches",
    2: "inclusion",
    3: "pitted_surface",
    4: "scratches",
    5: "rolled-in_scale"
}

CLASS_TO_ID = {
    name: class_id
    for class_id, name in CLASS_NAMES.items()
}

# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# READ VOC XML
# ============================================================

def load_ground_truth(xml_path):

    ground_truth = []

    if not os.path.exists(xml_path):
        return ground_truth

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):

        name_element = obj.find("name")

        if name_element is None:
            continue

        class_name = name_element.text.strip()

        if class_name not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[class_name]

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        ground_truth.append({
            "class_id": class_id,
            "class_name": class_name,
            "box": [xmin, ymin, xmax, ymax]
        })

    return ground_truth


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(stem):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]

    for ext in extensions:

        path = os.path.join(
            IMAGE_DIR,
            stem + ext
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# STATISTICS
# ============================================================

overall = {
    "TP": 0,
    "FP": 0,
    "FN": 0
}

class_stats = defaultdict(
    lambda: {
        "TP": 0,
        "FP": 0,
        "FN": 0
    }
)

total_images = 0
processed_images = 0

# ============================================================
# PROCESS ANNOTATIONS
# ============================================================

xml_files = [
    f for f in os.listdir(ANNOTATION_DIR)
    if f.lower().endswith(".xml")
]

total_images = len(xml_files)

print(f"\nAnnotations found : {total_images}")
print("Starting analysis...\n")

for index, xml_filename in enumerate(xml_files, start=1):

    stem = os.path.splitext(xml_filename)[0]

    xml_path = os.path.join(
        ANNOTATION_DIR,
        xml_filename
    )

    image_path = find_image(stem)

    if image_path is None:
        continue

    image = cv2.imread(image_path)

    if image is None:
        continue

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth(xml_path)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=416,
        verbose=False
    )

    result = results[0]

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            predictions.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    str(class_id)
                ),
                "confidence": confidence,
                "box": coordinates
            })

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matched_ground_truth = set()
    matched_predictions = set()

    # Match highest-confidence predictions first
    predictions_sorted = sorted(
        enumerate(predictions),
        key=lambda x: x[1]["confidence"],
        reverse=True
    )

    for pred_index, prediction in predictions_sorted:

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt in enumerate(ground_truth):

            if gt_index in matched_ground_truth:
                continue

            # Classes must match
            if prediction["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index >= 0
            and best_iou >= IOU_THRESHOLD
        ):

            # TRUE POSITIVE

            matched_predictions.add(pred_index)
            matched_ground_truth.add(best_gt_index)

            class_name = prediction["class_name"]

            overall["TP"] += 1
            class_stats[class_name]["TP"] += 1

    # --------------------------------------------------------
    # FALSE POSITIVES
    # --------------------------------------------------------

    for pred_index, prediction in enumerate(predictions):

        if pred_index in matched_predictions:
            continue

        class_name = prediction["class_name"]

        overall["FP"] += 1
        class_stats[class_name]["FP"] += 1

    # --------------------------------------------------------
    # FALSE NEGATIVES
    # --------------------------------------------------------

    for gt_index, gt in enumerate(ground_truth):

        if gt_index in matched_ground_truth:
            continue

        class_name = gt["class_name"]

        overall["FN"] += 1
        class_stats[class_name]["FN"] += 1

    processed_images += 1

    # Progress
    if index % 50 == 0 or index == total_images:

        print(
            f"Processed {index}/{total_images} images..."
        )


# ============================================================
# METRICS
# ============================================================

TP = overall["TP"]
FP = overall["FP"]
FN = overall["FN"]

if TP + FP > 0:
    precision = TP / (TP + FP)
else:
    precision = 0.0

if TP + FN > 0:
    recall = TP / (TP + FN)
else:
    recall = 0.0

if precision + recall > 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0.0


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL RESULTS")
print("-" * 60)

print(f"Images Processed : {processed_images}")
print(f"True Positives   : {TP}")
print(f"False Positives  : {FP}")
print(f"False Negatives  : {FN}")

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\n")
print("=" * 60)
print("CLASS-WISE ERROR ANALYSIS")
print("-" * 60)

for class_id in range(6):

    class_name = CLASS_NAMES[class_id]

    stats = class_stats[class_name]

    print(
        f"{class_name:<20}"
        f"TP={stats['TP']:5d} "
        f"FP={stats['FP']:5d} "
        f"FN={stats['FN']:5d}"
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Class",
        "True Positives",
        "False Positives",
        "False Negatives"
    ])

    for class_id in range(6):

        class_name = CLASS_NAMES[class_id]

        stats = class_stats[class_name]

        writer.writerow([
            class_name,
            stats["TP"],
            stats["FP"],
            stats["FN"]
        ])

    writer.writerow([])

    writer.writerow([
        "OVERALL",
        TP,
        FP,
        FN
    ])

    writer.writerow([
        "Precision",
        precision
    ])

    writer.writerow([
        "Recall",
        recall
    ])

    writer.writerow([
        "F1 Score",
        f1
    ])


print("\nResults saved to:")
print(OUTPUT_CSV)

print("\nAnalysis completed.")