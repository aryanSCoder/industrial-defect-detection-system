from ultralytics import YOLO
import os
import csv
from datetime import datetime

MODEL_PATH = r"models\best.pt"

print("=" * 60)
print("INDUSTRIAL DEFECT DETECTION")
print("HYPERPARAMETER TUNING")
print("=" * 60)

model = YOLO(MODEL_PATH)

experiments = [
    {
        "name": "baseline",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "batch": 8
    },
    {
        "name": "low_lr",
        "lr0": 0.005,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "batch": 8
    },
    {
        "name": "high_lr",
        "lr0": 0.02,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "batch": 8
    },
    {
        "name": "higher_weight_decay",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.001,
        "batch": 8
    }
]

results = []

for exp in experiments:

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENT: {exp['name']}")
    print("=" * 60)

    try:

        run_name = f"tuning_{exp['name']}"

        result = model.train(
            data=r"dataset\NEU-DET\data.yaml",
            epochs=10,
            imgsz=416,
            batch=exp["batch"],
            lr0=exp["lr0"],
            momentum=exp["momentum"],
            weight_decay=exp["weight_decay"],
            project="runs/tuning",
            name=run_name,
            exist_ok=True,
            verbose=True
        )

        metrics = model.val(
            data=r"dataset\NEU-DET\data.yaml",
            imgsz=416,
            batch=exp["batch"],
            verbose=False
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        results.append({
            "experiment": exp["name"],
            "learning_rate": exp["lr0"],
            "momentum": exp["momentum"],
            "weight_decay": exp["weight_decay"],
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map5095
        })

        print("\nRESULT:")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"mAP50     : {map50:.4f}")
        print(f"mAP50-95  : {map5095:.4f}")

    except Exception as e:

        print(f"\nExperiment failed: {e}")

        results.append({
            "experiment": exp["name"],
            "learning_rate": exp["lr0"],
            "momentum": exp["momentum"],
            "weight_decay": exp["weight_decay"],
            "precision": "ERROR",
            "recall": "ERROR",
            "mAP50": "ERROR",
            "mAP50-95": "ERROR"
        })


os.makedirs("runs/tuning", exist_ok=True)

csv_path = "runs/tuning/hyperparameter_results.csv"

with open(csv_path, "w", newline="") as file:

    fieldnames = [
        "experiment",
        "learning_rate",
        "momentum",
        "weight_decay",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for row in results:
        writer.writerow(row)


print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 60)

print(f"\nResults saved to:")
print(csv_path)

print("\nExperiment Summary:")

for row in results:
    print(
        f"{row['experiment']:25} "
        f"mAP50={row['mAP50']} "
        f"mAP50-95={row['mAP50-95']}"
    )

print("\nCompleted:", datetime.now())