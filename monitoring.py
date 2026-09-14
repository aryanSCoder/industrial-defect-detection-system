from flask import Flask, jsonify
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
import random
import time

app = Flask(__name__)

# Metrics
detections_total = Counter(
    "industrial_detections_total",
    "Total number of detections"
)

defects_total = Counter(
    "industrial_defects_total",
    "Total defects detected",
    ["defect_type"]
)

detection_latency = Histogram(
    "industrial_detection_latency_seconds",
    "Detection processing latency"
)

camera_status = Gauge(
    "industrial_camera_status",
    "Camera status: 1=online, 0=offline"
)

plc_status = Gauge(
    "industrial_plc_status",
    "PLC status: 1=connected, 0=disconnected"
)

current_defect = Gauge(
    "industrial_current_defect",
    "Current defect detected: 1=yes, 0=no"
)

# Initial status
camera_status.set(1)
plc_status.set(1)
current_defect.set(0)


@app.route("/")
def home():
    return jsonify({
        "system": "Industrial Defect Detection Monitoring",
        "status": "running"
    })


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


@app.route("/test_detection", methods=["GET", "POST"])
def test_detection():

    start_time = time.time()

    defect_types = [
        "crazing",
        "patches",
        "inclusion",
        "pitted_surface",
        "scratches",
        "rolled-in_scale"
    ]

    defect = random.choice(defect_types)
    confidence = round(random.uniform(60, 98), 2)

    detections_total.inc()
    defects_total.labels(defect_type=defect).inc()
    current_defect.set(1)

    detection_latency.observe(time.time() - start_time)

    return jsonify({
        "status": "success",
        "defect_detected": True,
        "defect_type": defect,
        "confidence": confidence
    })


@app.route("/camera/<status>", methods=["GET"])
def camera(status):

    if status.lower() == "online":
        camera_status.set(1)
    else:
        camera_status.set(0)

    return jsonify({
        "camera_status": status
    })


@app.route("/plc/<status>", methods=["GET"])
def plc(status):

    if status.lower() == "connected":
        plc_status.set(1)
    else:
        plc_status.set(0)

    return jsonify({
        "plc_status": status
    })


if __name__ == "__main__":
    print("Industrial Monitoring Server")
    print("Running on http://127.0.0.1:8000")

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )