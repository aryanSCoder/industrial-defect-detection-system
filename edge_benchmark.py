import os
import time
import csv
import cv2
import numpy as np
import onnxruntime as ort

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\best.onnx"
IMAGE_PATH = r"dataset\NEU-DET\IMAGES\crazing_1.jpg"

IMG_SIZE = 416
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

OUTPUT_DIR = r"runs\edge_benchmark"
CSV_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("ONNX EDGE PERFORMANCE BENCHMARK")
print("=" * 60)

print("\nLoading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully.")
print("Execution Provider : CPUExecutionProvider")
print("Input Name         :", input_name)
print("Input Size         :", IMG_SIZE, "x", IMG_SIZE)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )

print("\nTest image loaded:", IMAGE_PATH)


# ============================================================
# PREPROCESSING
# ============================================================

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized = cv2.resize(
    image_rgb,
    (IMG_SIZE, IMG_SIZE)
)

input_tensor = resized.astype(np.float32) / 255.0

input_tensor = np.transpose(
    input_tensor,
    (2, 0, 1)
)

input_tensor = np.expand_dims(
    input_tensor,
    axis=0
)

print("Input tensor shape:", input_tensor.shape)


# ============================================================
# WARM-UP
# ============================================================

print("\nRunning warm-up...")

for _ in range(WARMUP_RUNS):
    session.run(
        None,
        {input_name: input_tensor}
    )

print("Warm-up completed.")


# ============================================================
# BENCHMARK
# ============================================================

print("\nRunning benchmark...")
print("Benchmark runs:", BENCHMARK_RUNS)

latencies = []

for i in range(BENCHMARK_RUNS):

    start_time = time.perf_counter()

    session.run(
        None,
        {input_name: input_tensor}
    )

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    latencies.append(latency_ms)

    if (i + 1) % 10 == 0:
        print(
            f"Completed {i + 1}/{BENCHMARK_RUNS} runs"
        )


# ============================================================
# CALCULATE RESULTS
# ============================================================

average_latency = np.mean(latencies)
minimum_latency = np.min(latencies)
maximum_latency = np.max(latencies)
median_latency = np.median(latencies)

fps = 1000 / average_latency


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(f"Average Latency : {average_latency:.2f} ms")
print(f"Minimum Latency : {minimum_latency:.2f} ms")
print(f"Maximum Latency : {maximum_latency:.2f} ms")
print(f"Median Latency  : {median_latency:.2f} ms")
print(f"Estimated FPS   : {fps:.2f}")

print("=" * 60)


# ============================================================
# SAVE CSV
# ============================================================

results = [
    {
        "model": "YOLOv8 ONNX",
        "execution_provider": "CPU",
        "image_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "benchmark_runs": BENCHMARK_RUNS,
        "average_latency_ms": round(
            float(average_latency), 2
        ),
        "minimum_latency_ms": round(
            float(minimum_latency), 2
        ),
        "maximum_latency_ms": round(
            float(maximum_latency), 2
        ),
        "median_latency_ms": round(
            float(median_latency), 2
        ),
        "estimated_fps": round(
            float(fps), 2
        )
    }
]

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


print("\nResults saved to:")
print(CSV_PATH)

print("\nBenchmark completed successfully.")